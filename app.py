import os
import io
import json
from typing import List, Dict

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from openai import OpenAI

# ---- UI ----
st.set_page_config(page_title="DocuChat++", page_icon="📚", layout="wide")
st.title("📚 DocuChat++ — PDF Q&A with Retrieval + Trained QA Model")

with st.sidebar:
    st.header("Upload PDFs")
    files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )
    ingest = st.button("🔎 Ingest / Rebuild Index")
    show_debug = st.checkbox("Show retrieved passages (debug)", value=False)

    st.markdown("---")
    answer_mode = st.selectbox(
        "Answer mode",
        options=["Local QA model", "GPT (context only)", "Both"],
        index=0,
        help="Choose whether to answer using your fine-tuned QA model, GPT with context, or both.",
    )

# ---- API key / client for GPT ----
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.warning("OPENAI_API_KEY not set. GPT mode will be disabled.")
    client = None
else:
    client = OpenAI(api_key=API_KEY)

# ---- Globals / cache ----
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


EMB = get_embedder()
DIM = EMB.get_sentence_embedding_dimension()

INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.npy")

MODELS_DIR = "models"
RESULTS_JSON = os.path.join(MODELS_DIR, "results.json")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for p in pdf.pages:
        text += (p.extract_text() or "")
    return text


def chunk_text(text: str, chunk_size=750, overlap=100):
    words = text.split()
    i, chunks = 0, []
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunks.append(" ".join(words[i:j]))
        if j == len(words):
            break
        i = j - overlap
        if i < 0:
            i = 0
    return chunks


def build_index(docs: List[Dict[str, str]]):
    """docs = [{'text': str, 'source': str}]"""
    texts = [d["text"] for d in docs]
    if not texts:
        return None, []
    X = EMB.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(DIM)
    faiss.normalize_L2(X)
    index.add(np.array(X, dtype="float32"))
    # store (source, text)
    np.save(
        META_PATH,
        np.array([(d["source"], t) for d, t in zip(docs, texts)], dtype=object),
    )
    faiss.write_index(index, INDEX_PATH)
    return index, texts


def load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return None, []
    index = faiss.read_index(INDEX_PATH)
    meta = np.load(META_PATH, allow_pickle=True)
    texts = [row[1] for row in meta]
    return index, texts


def search(query: str, k=4):
    index, texts = load_index()
    if index is None:
        return []
    q = EMB.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q, dtype="float32"), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        results.append({"text": texts[int(idx)], "score": float(score)})
    return results


# ---- Load best QA model (from training) ----
@st.cache_resource(show_spinner=True)
def load_best_qa_model():
    """
    Load the best QA model based on models/results.json.
    Returns (tokenizer, model, model_name) or (None, None, None) if not available.
    """
    if not os.path.exists(RESULTS_JSON):
        st.warning(
            "No results.json found in 'models/'. "
            "Run ml/train_qa_models.py first to train QA models."
        )
        return None, None, None

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_model_name = data.get("best_model")
    if not best_model_name:
        st.warning("results.json does not contain 'best_model'.")
        return None, None, None

    model_dir = os.path.join(MODELS_DIR, best_model_name)
    if not os.path.exists(model_dir):
        st.warning(f"Best model directory not found: {model_dir}")
        return None, None, None

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return tokenizer, model, best_model_name


QA_TOKENIZER, QA_MODEL, QA_MODEL_NAME = load_best_qa_model()


def answer_with_local_qa(question: str, context: str) -> str:
    """
    Use the fine-tuned QA model to answer a question given context.
    Returns the extracted answer text.
    """
    if QA_MODEL is None or QA_TOKENIZER is None:
        return "Local QA model not available. Please run training first."

    device = next(QA_MODEL.parameters()).device

    inputs = QA_TOKENIZER(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = QA_MODEL(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_index = int(torch.argmax(start_logits))
        end_index = int(torch.argmax(end_logits))

    if end_index < start_index:
        end_index = start_index

    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[start_index : end_index + 1]
    answer_text = QA_TOKENIZER.decode(answer_tokens, skip_special_tokens=True)

    return answer_text.strip()


def answer_with_gpt(question: str, hits: List[Dict[str, str]]) -> str:
    """
    Use GPT with retrieved context only.
    """
    if client is None:
        return "GPT mode is disabled because OPENAI_API_KEY is not set."

    if not hits:
        return "No context retrieved for GPT."

    ctx = "\n\n".join([f"[{i+1}] {h['text']}" for i, h in enumerate(hits)])
    prompt = f"""Answer using ONLY the context below. If the answer is not present, say you don't know.
Question: {question}

Context:
{ctx}

Return the answer and cite passages like [1], [2].
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You answer strictly from the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        # Do NOT crash the whole app during your presentation.
        return f"GPT error: {e}. Please check your OPENAI_API_KEY."

# ---- Ingest PDFs ----
if files and ingest:
    st.info("Indexing… this may take a minute on first run.")
    docs = []
    for f in files:
        text = extract_text_from_pdf(f.read())
        for ch in chunk_text(text):
            docs.append({"text": ch, "source": f.name})
    index, _ = build_index(docs)
    if index:
        st.success(f"Indexed {len(docs)} chunks from {len(files)} file(s).")

# ---- Ask ----
st.subheader("Ask a question")
query = st.text_input("Type your question")
top_k = st.slider("Top-K passages", 1, 10, 4)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        hits = search(query, k=top_k)
        if not hits:
            st.warning("No index yet or no results. Upload PDFs and click Ingest first.")
        else:
            # Debug view of retrieved passages
            if show_debug:
                st.markdown("### Top Passages (debug)")
                for i, h in enumerate(hits, start=1):
                    st.markdown(
                        f"**[{i}] score={h['score']:.3f}**\n\n"
                        f"> {h['text'][:600]}{'…' if len(h['text'])>600 else ''}"
                    )

            # Build context from retrieved passages
            context = "\n\n".join([h["text"] for h in hits])

            st.markdown("### Answers")

            if answer_mode in ["Local QA model", "Both"]:
                with st.spinner("Running local QA model..."):
                    local_answer = answer_with_local_qa(query, context)
                st.markdown(f"**Local QA model ({QA_MODEL_NAME or 'N/A'}):**")
                st.write(local_answer)

            if answer_mode in ["GPT (context only)", "Both"]:
                with st.spinner("Querying GPT with retrieved context..."):
                    gpt_answer = answer_with_gpt(query, hits)
                st.markdown("**GPT (context-only answer):**")
                st.write(gpt_answer)
