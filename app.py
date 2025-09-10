# app.py — StudyMate (Hackathon-ready full app)
# Requirements: streamlit, pymupdf (fitz), faiss-cpu, numpy, google-generativeai
# Optional but strongly recommended to avoid Gemini embedding quotas: sentence-transformers

import os
import re
import json
import csv
import base64
from io import BytesIO, StringIO
from typing import List, Dict, Tuple, Any

import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import google.generativeai as genai

# ---------------------------
# CONFIGURE GEMINI (PASTE YOUR KEY BELOW)
# ---------------------------
# Replace "YOUR_GEMINI_KEY" with your actual Gemini API key (keep the quotes)
# Example: genai.configure(api_key="AIzaSyDUMMY_EXAMPLE_KEY")
genai.configure(api_key="AIzaSyBxWIr26jdERhwckvMO_mFUrbpbfi49_YA")  # <<< PASTE YOUR GEMINI KEY HERE

EMBED_MODEL = "models/embedding-001"
ANSWER_MODEL = "models/gemini-1.5-flash"

# ---------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="StudyMate", layout="wide", initial_sidebar_state="expanded")
st.title("📘 StudyMate — Multi-PDF Smart Study Tool")
st.markdown("Upload multiple PDFs, ask questions, generate quizzes & flashcards, generate summaries, and export everything (TXT + CSV).")

# ---------------------------
# UTILS: text extraction, chunking
# ---------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract plain text from a PDF file using PyMuPDF (fitz)."""
    try:
        uploaded_file.seek(0)
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"PDF read error: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into word chunks with overlap."""
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - overlap)
    return chunks

# ---------------------------
# EMBEDDING: prefer local SentenceTransformer to avoid Gemini quota errors.
# ---------------------------

from sentence_transformers import SentenceTransformer

# Load MiniLM once into session_state
if "local_embed_model" not in st.session_state:
    try:
        st.session_state.local_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"Could not load local embedding model: {e}")
        st.session_state.local_embed_model = None

def safe_embed(texts: List[str]) -> np.ndarray:
    """
    Produce embeddings for texts.
    Primary method: local SentenceTransformer (all-MiniLM-L6-v2, 384 dims).
    Fallback: Gemini genai.embed_content (quota-limited).
    Returns numpy float32 array.
    """
    if not texts:
        return np.zeros((0, 384), dtype="float32")

    # ✅ Local MiniLM
    if st.session_state.local_embed_model:
        try:
            embs = st.session_state.local_embed_model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
            return np.array(embs, dtype="float32")
        except Exception as e:
            st.warning(f"Local embedding error: {e}")

    # ⚠️ Fallback: Gemini embeddings (will consume quota)
    embs = []
    for t in texts:
        try:
            r = genai.embed_content(model=EMBED_MODEL, content=t)
            emb = r.get("embedding") if isinstance(r, dict) else r["embedding"]
            embs.append(emb)
        except Exception as e:
            st.warning(f"Gemini embedding failed: {e}")
            embs.append([0.0] * 1536)  # fallback vector
    return np.array(embs, dtype="float32")

# ---------------------------
# FAISS index helpers
# ---------------------------
def build_faiss_index(chunks: List[str]):
    """Build a FAISS IndexFlatL2 index for a list of chunk strings."""
    if not chunks:
        return None, None
    embeddings = safe_embed(chunks)
    if embeddings.size == 0:
        return None, None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    try:
        index.add(embeddings)
    except Exception as e:
        st.warning(f"FAISS add failed: {e}")
        return None, embeddings
    return index, embeddings

def retrieve_from_index(question: str, index, chunks: List[str], k: int = 5) -> Tuple[str, List[int]]:
    """
    Search index for question, build context from top chunks, and call Gemini to answer.
    Returns (answer_text, list_of_top_chunk_indices).
    """
    if index is None:
        return "[Index unavailable]", []
    try:
        q_emb = safe_embed([question])
        D, I = index.search(q_emb, k=k)
        top_idx = [int(x) for x in I[0] if int(x) >= 0]
        # Compose short context from top 3 chunks (for grounding)
        context_chunks = [chunks[i] for i in top_idx[:3] if i < len(chunks)]
        context = "\n\n".join(context_chunks)
        # Prompt instructs the model to answer from context only
        prompt = f"Answer the question based ONLY on the context below. If the context doesn't contain the answer, say you cannot answer.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer concisely for a student."
        model = genai.GenerativeModel(ANSWER_MODEL)
        resp = model.generate_content(prompt)
        return resp.text, top_idx
    except Exception as e:
        return f"[Error generating answer: {e}]", []
# ---------------------------
# EXPORT helpers: TXT + CSV (reliable)
# ---------------------------
def chat_to_txt(chat_history: List[Tuple[str,str,List[Tuple[int,str]]]]) -> bytes:
    s = StringIO()
    for i, (q,a,refs) in enumerate(chat_history, start=1):
        s.write(f"Q{i}: {q}\nA{i}: {a}\nSources:\n")
        for pid, short in refs:
            s.write(f" - PDF {pid+1}: {short}\n")
        s.write("\n" + ("-"*50) + "\n")
    return s.getvalue().encode("utf-8")

def chat_to_csv(chat_history: List[Tuple[str,str,List[Tuple[int,str]]]]) -> bytes:
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["Question","Answer","Sources"])
    for q,a,refs in chat_history:
        sources = " | ".join([f"PDF{pid+1}:{short}" for pid,short in refs])
        w.writerow([q,a,sources])
    return buf.getvalue().encode("utf-8")

def summaries_to_txt(summaries: Dict[int,str]) -> bytes:
    buf = StringIO()
    for pid, text in summaries.items():
        buf.write(f"--- Summary PDF {pid+1} ---\n{text}\n\n")
    return buf.getvalue().encode("utf-8")

def summaries_to_csv(summaries: Dict[int,str]) -> bytes:
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["PDF Index","Summary"])
    for pid, text in summaries.items():
        w.writerow([pid+1, text])
    return buf.getvalue().encode("utf-8")

def quiz_to_txt(quiz_dict: Dict[int, Dict]) -> bytes:
    buf = StringIO()
    for pid, quiz in quiz_dict.items():
        buf.write(f"--- Quiz for PDF {pid+1} ---\n")
        mcqs = quiz.get("mcqs", [])
        for i, mcq in enumerate(mcqs, start=1):
            buf.write(f"Q{i}: {mcq.get('question','')}\n")
            for j,opt in enumerate(mcq.get("options",[]), start=1):
                mark = "*" if j-1 == mcq.get("answer",0) else "-"
                buf.write(f"  {mark} {opt}\n")
            buf.write("\n")
        buf.write("Flashcards:\n")
        for i,fc in enumerate(quiz.get("flashcards",[]), start=1):
            buf.write(f"F{i}: {fc.get('q','')} => {fc.get('a','')}\n")
        buf.write("\n")
    return buf.getvalue().encode("utf-8")

def quiz_to_csv(quiz_dict: Dict[int, Dict]) -> bytes:
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["PDF Index","Type","Question","Options","Answer"])
    for pid, quiz in quiz_dict.items():
        for mcq in quiz.get("mcqs", []):
            opts = " || ".join(mcq.get("options",[]))
            ans = mcq.get("options",[])[mcq.get("answer",0)] if mcq.get("options") else ""
            w.writerow([pid+1,"MCQ", mcq.get("question",""), opts, ans])
        for fc in quiz.get("flashcards", []):
            w.writerow([pid+1,"Flashcard", fc.get("q",""), fc.get("a",""), ""])
    return buf.getvalue().encode("utf-8")

def flashcards_to_txt(quiz_dict: Dict[int, Dict]) -> bytes:
    buf = StringIO()
    for pid, quiz in quiz_dict.items():
        flashcards = quiz.get("flashcards", [])
        if not flashcards:
            continue
        buf.write(f"--- Flashcards for PDF {pid+1} ---\n")
        for i,fc in enumerate(flashcards, start=1):
            buf.write(f"F{i}: {fc.get('q','')} => {fc.get('a','')}\n")
        buf.write("\n")
    return buf.getvalue().encode("utf-8")

def flashcards_to_csv(quiz_dict: Dict[int, Dict]) -> bytes:
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["PDF Index","Question","Answer"])
    for pid, quiz in quiz_dict.items():
        for fc in quiz.get("flashcards", []):
            w.writerow([pid+1, fc.get("q",""), fc.get("a","")])
    return buf.getvalue().encode("utf-8")

def everything_to_txt(chat_history, summaries, quizzes) -> bytes:
    s = StringIO()
    s.write("=== CHAT ===\n\n")
    s.write(chat_to_txt(chat_history).decode("utf-8"))
    s.write("\n=== SUMMARIES ===\n\n")
    s.write(summaries_to_txt(summaries).decode("utf-8"))
    s.write("\n=== QUIZZES ===\n\n")
    s.write(quiz_to_txt(quizzes).decode("utf-8"))
    return s.getvalue().encode("utf-8")

def everything_to_csv(chat_history, summaries, quizzes) -> bytes:
    s = StringIO()
    s.write("--- CHAT ---\n")
    s.write(chat_to_csv(chat_history).decode("utf-8"))
    s.write("\n--- SUMMARIES ---\n")
    s.write(summaries_to_csv(summaries).decode("utf-8"))
    s.write("\n--- QUIZZES ---\n")
    s.write(quiz_to_csv(quizzes).decode("utf-8"))
    return s.getvalue().encode("utf-8")

# ---------------------------
# PROMPTS: quiz generation (Gemini)
# ---------------------------
def ask_gemini_for_quiz(text: str, n_mcq:int=6, n_flash:int=6) -> Dict[str, Any]:
    prompt = f"""
You MUST return valid JSON only.
From the document below create:
1) "mcqs": an array of {n_mcq} multiple-choice questions. Each object must have:
   - "question": string
   - "options": array of 4 strings
   - "answer": integer (0-3) index of correct option
2) "flashcards": an array of {n_flash} objects with "q" and "a".

Return ONLY valid JSON. Document:
{text}
"""
    try:
        model = genai.GenerativeModel(ANSWER_MODEL)
        resp = model.generate_content(prompt)
        txt = resp.text.strip()
        # try direct parse
        try:
            return json.loads(txt)
        except Exception:
            # try to extract JSON substring
            s = txt.find("{")
            e = txt.rfind("}")
            if s != -1 and e != -1:
                return json.loads(txt[s:e+1])
            return {"error": txt}
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Sidebar: Appearance (background image)
# ---------------------------
st.sidebar.header("Appearance")
bg_file = st.sidebar.file_uploader("Upload background image (optional)", type=["png","jpg","jpeg"])
if bg_file:
    try:
        data = bg_file.read()
        b64 = base64.b64encode(data).decode()
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{background-image: url("data:image/png;base64,{b64}"); background-size: cover; background-position: center;}}
            .stButton>button {{font-weight:600}}
            .stApp {{color: #fff}}
            </style>
            """, unsafe_allow_html=True
        )
    except Exception:
        st.warning("Couldn't load background image; using default.")
# ---------------------------
# Upload PDFs (multiple)
# ---------------------------
uploaded_files = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more PDFs to start. For demo: upload 1–3 PDFs (or more).")
    st.stop()

# ---------------------------
# Extract text & chunk for each PDF
# ---------------------------
st.success(f"{len(uploaded_files)} PDF(s) uploaded.")
pdf_texts: List[str] = []
per_pdf_chunks: List[List[str]] = []
per_pdf_indices: List[Any] = []  # each element is FAISS index or None
per_pdf_embeddings: List[Any] = []

with st.spinner("Extracting and chunking PDFs..."):
    for idx, f in enumerate(uploaded_files):
        txt = extract_text_from_pdf(f)
        pdf_texts.append(txt)
        chunks = chunk_text(txt)
        per_pdf_chunks.append(chunks)
        st.write(f"**{f.name}** — {len(chunks)} chunks")

# Display chunk expanders grouped per PDF
for pid, chunks in enumerate(per_pdf_chunks):
    with st.expander(f"📄 {uploaded_files[pid].name} — View chunks ({len(chunks)})"):
        for i, c in enumerate(chunks):
            short = c[:300].replace("\n", " ")
            st.markdown(f"**Chunk {i+1}:** {short}...")

# ---------------------------
# Build FAISS indices per PDF (separate)
# ---------------------------
with st.spinner("Building FAISS indexes (per PDF)..."):
    for pid, chunks in enumerate(per_pdf_chunks):
        if len(chunks) == 0:
            per_pdf_indices.append(None)
            per_pdf_embeddings.append(None)
            continue
        idx_obj, emb = build_faiss_index(chunks)
        per_pdf_indices.append(idx_obj)
        per_pdf_embeddings.append(emb)

st.success("Indexes built. You can now ask questions, generate quizzes & flashcards, and export results.")

# ---------------------------
# Session state initialization
# ---------------------------
if "chat_history" not in st.session_state:
    # chat_history stores tuples: (question_display, answer_text, [(pdf_idx, short_ref), ...])
    st.session_state.chat_history = []
if "summaries" not in st.session_state:
    st.session_state.summaries = {}  # pid -> summary_text
if "quizzes" not in st.session_state:
    st.session_state.quizzes = {}  # pid -> quiz dict
if "mcq_keys" not in st.session_state:
    st.session_state.mcq_keys = {}  # ui keys per mcq selection
if "flashcard_keys" not in st.session_state:
    st.session_state.flashcard_keys = {}

# ---------------------------
# Q&A Section (with PDF targeting)
# ---------------------------
st.header("💬 Ask Questions from PDFs")
st.markdown("Tip: include `PDF 1` (or `PDF 2`) in your question to target a specific file. Otherwise StudyMate will pick the best-matching PDF automatically.")

question = st.text_input("Ask a question (searched intelligently across PDFs)", placeholder="e.g. 'Explain briefly about PDF 1'")

if st.button("🔍 Get Answer") and question.strip():
    # check if user explicitly targeted a PDF like "pdf 2"
    m = re.search(r"pdf\s*(\d+)", question.lower())
    if m:
        target = int(m.group(1)) - 1
        if 0 <= target < len(per_pdf_indices) and per_pdf_indices[target] is not None:
            ans, top_idx = retrieve_from_index(question, per_pdf_indices[target], per_pdf_chunks[target], k=5)
            refs = []
            for ti in top_idx:
                if ti < len(per_pdf_chunks[target]):
                    short = per_pdf_chunks[target][ti][:250].replace("\n", " ")
                    refs.append((target, short))
            display_q = f"[PDF {target+1}] {question}"
            st.session_state.chat_history.append((display_q, ans, refs))
            st.subheader(f"📘 Answer from PDF {target+1}: {uploaded_files[target].name}")
            st.write(ans)
            if refs:
                st.markdown("**Reference (short):**")
                for _, short in refs:
                    st.info(short)
        else:
            st.error("Requested PDF number not available or not indexed.")
    else:
        # automatic selection: compute 1-NN distance across indexes & pick best match
        best_pdf = None
        best_dist = float("inf")
        best_ans = None
        best_refs = []
        # compute single question embedding once for speed
        try:
            q_emb = safe_embed([question])
            for pid, idx_obj in enumerate(per_pdf_indices):
                if idx_obj is None:
                    continue
                try:
                    D, I = idx_obj.search(q_emb, k=1)
                    dist = float(D[0][0])
                    if dist < best_dist:
                        # get full answer for this PDF using top-k retrieval
                        ans, top_idx = retrieve_from_index(question, idx_obj, per_pdf_chunks[pid], k=5)
                        refs = []
                        for ti in top_idx:
                            if ti < len(per_pdf_chunks[pid]):
                                short = per_pdf_chunks[pid][ti][:250].replace("\n", " ")
                                refs.append((pid, short))
                        best_pdf, best_dist, best_ans, best_refs = pid, dist, ans, refs
                except Exception:
                    continue
        except Exception as e:
            st.error(f"Embedding error for query: {e}")
            best_pdf = None

        if best_pdf is not None:
            display_q = f"[PDF {best_pdf+1}] {question}"
            st.session_state.chat_history.append((display_q, best_ans, best_refs))
            st.subheader(f"📘 Answer from PDF {best_pdf+1}: {uploaded_files[best_pdf].name}")
            st.write(best_ans)
            if best_refs:
                st.markdown("**Reference (short):**")
                for _, short in best_refs:
                    st.info(short)
        else:
            st.error("Couldn't find a relevant PDF index or embeddings failed. Try a different question.")

# Chat history — kept hidden by default in an expander
with st.expander("📜 View Chat History (click to open)"):

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑 Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

    # show history items (most recent last)
    if st.session_state.chat_history:
        for i, (q, a, refs) in enumerate(reversed(st.session_state.chat_history), start=1):
            st.markdown(f"**Q{i}:** {q}")
            st.write(a)
            if refs:
                st.markdown("**Sources:**")
                for pid, short in refs:
                    st.info(f"{uploaded_files[pid].name}: {short}")
            st.markdown("---")
    else:
        st.info("No chat history yet.")
# ---------------------------
# Summaries (per PDF)
# ---------------------------
st.header("📝 Summaries (per PDF)")
col_s1, col_s2 = st.columns([1, 1])
with col_s1:
    if st.button("Generate summaries for ALL PDFs"):
        # generate summaries and store in session
        for pid, txt in enumerate(pdf_texts):
            if not txt.strip():
                st.session_state.summaries[pid] = "No text extracted."
                continue
            try:
                model = genai.GenerativeModel(ANSWER_MODEL)
                resp = model.generate_content(f"Summarize this document concisely for a student:\n\n{txt}")
                st.session_state.summaries[pid] = resp.text
            except Exception as e:
                st.session_state.summaries[pid] = f"[Summary failed: {e}]"
        st.success("Summaries generated.")
with col_s2:
    if st.button("Clear summaries"):
        st.session_state.summaries = {}
        st.success("Summaries cleared.")

# show each summary in its own expander
for pid, s in st.session_state.summaries.items():
    with st.expander(f"Summary — {uploaded_files[pid].name}", expanded=False):
        st.write(s)

# ---------------------------
# Quiz Generator & Flashcards (per PDF)
# ---------------------------
st.header("🎯 Quiz Generator (per PDF)")

num_mcq = st.number_input("MCQs per PDF (4–12)", min_value=4, max_value=12, value=6, step=1)
num_flash = st.number_input("Flashcards per PDF (4–12)", min_value=4, max_value=12, value=6, step=1)

col_q1, col_q2 = st.columns([1, 1])
with col_q1:
    if st.button("Generate Quizzes and Flashcards for ALL PDFs"):
        for pid, txt in enumerate(pdf_texts):
            if not txt.strip():
                st.session_state.quizzes[pid] = {"mcqs": [], "flashcards": []}
                continue
            parsed = ask_gemini_for_quiz(txt, n_mcq=num_mcq, n_flash=num_flash)
            st.session_state.quizzes[pid] = parsed
        st.success("Quizzes & flashcards generated.")
with col_q2:
    if st.button("Clear Quizzes and Flashcards"):
        st.session_state.quizzes = {}
        st.success("Quizzes and flashcards cleared.")

# show quizzes interactive per PDF
for pid, quiz in st.session_state.quizzes.items():
    st.subheader(f"Quiz — {uploaded_files[pid].name}")
    if not quiz:
        st.info("No quiz generated for this PDF.")
        continue
    if "error" in quiz:
        st.error(f"Quiz generation error for PDF {pid+1}: {quiz['error']}")
        continue
    mcqs = quiz.get("mcqs", [])
    flashcards = quiz.get("flashcards", [])

    if mcqs:
        st.markdown("#### 📝 Multiple Choice Questions")
        for j, item in enumerate(mcqs):
            q_text = item.get("question", "No question text")
            options = item.get("options", [])
            correct_index = int(item.get("answer", 0)) if "answer" in item else 0

            # display question text
            st.markdown(f"**Q{j+1}: {q_text}**")
            display_options = ["-- Select an option --"] + options
            sel_key = f"pdf{pid}_mcq_sel_{j}"
            if sel_key not in st.session_state:
                st.session_state[sel_key] = display_options[0]

            # selectbox uses the placeholder as default so user must actively choose
            choice = st.selectbox(f"Choose answer for Q{j+1} (PDF {pid+1})", display_options, key=sel_key, index=0)

            check_key = f"pdf{pid}_mcq_check_{j}"
            if st.button(f"✅ Check Answer Q{j+1}", key=check_key):
                if choice == display_options[0]:
                    st.warning("Please select an option before checking.")
                else:
                    chosen_index = display_options.index(choice) - 1
                    if chosen_index == correct_index:
                        st.success("🎉 Correct!")
                    else:
                        correct_text = options[correct_index] if 0 <= correct_index < len(options) else "N/A"
                        st.error(f"❌ Wrong. Correct answer: {correct_text}")
    else:
        st.info("No MCQs generated for this PDF yet.")

    if flashcards:
        st.markdown("#### 🎴 Flashcards")
        for j, card in enumerate(flashcards, start=1):
            qf = card.get("q", f"Flashcard {j}")
            af = card.get("a", "No answer provided.")
            with st.expander(f"Flashcard {j}: {qf}"):
                st.write(af)

# ---------------------------
# Downloads / Exports (TXT + CSV)
# ---------------------------
st.header("⬇️ Downloads (TXT + CSV)")

# Chat downloads
if st.session_state.chat_history:
    st.download_button("💾 Download Chat (TXT)", data=chat_to_txt(st.session_state.chat_history), file_name="chat_history.txt", mime="text/plain")
    st.download_button("💾 Download Chat (CSV)", data=chat_to_csv(st.session_state.chat_history), file_name="chat_history.csv", mime="text/csv")
else:
    st.info("No chat to download yet — ask some questions first.")

# Summaries downloads
if st.session_state.summaries:
    st.download_button("💾 Download Summaries (TXT)", data=summaries_to_txt(st.session_state.summaries), file_name="summaries.txt")
    st.download_button("💾 Download Summaries (CSV)", data=summaries_to_csv(st.session_state.summaries), file_name="summaries.csv")

# Quizzes downloads (MCQs + Flashcards combined)
if st.session_state.quizzes:
    st.download_button("💾 Download Quizzes (TXT)", data=quiz_to_txt(st.session_state.quizzes), file_name="quizzes.txt")
    st.download_button("💾 Download Quizzes (CSV)", data=quiz_to_csv(st.session_state.quizzes), file_name="quizzes.csv")
    # Flashcards-only downloads
    st.download_button("💾 Download Flashcards (TXT)", data=flashcards_to_txt(st.session_state.quizzes), file_name="flashcards.txt")
    st.download_button("💾 Download Flashcards (CSV)", data=flashcards_to_csv(st.session_state.quizzes), file_name="flashcards.csv")
else:
    st.info("No quizzes/flashcards to download yet. Generate quizzes first.")

# One-Click export everything both formats (creates two separate files)
col_all_a, col_all_b = st.columns([1,1])
with col_all_a:
    if st.button("One-Click: Export Everything (TXT)"):

        all_txt = everything_to_txt(st.session_state.chat_history, st.session_state.summaries, st.session_state.quizzes)
        st.download_button("Download EVERYTHING (TXT)", data=all_txt, file_name="studymate_everything.txt")
with col_all_b:
    if st.button("One-Click: Export Everything (CSV)"):
        all_csv = everything_to_csv(st.session_state.chat_history, st.session_state.summaries, st.session_state.quizzes)
        st.download_button("Download EVERYTHING (CSV)", data=all_csv, file_name="studymate_everything.csv")

# ---------------------------
# Footer / credits
# ---------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: green;'>❤️ Made with love by Ahmed</h4>", unsafe_allow_html=True)

# ---------------------------
# End of app.py
# ---------------------------
