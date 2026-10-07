import os
import random
import re
import shutil

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image
from pypdf import PdfReader
from pytesseract import Output
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StudyNotes AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 StudyNotes AI")
st.subheader("Ask questions from your lecture notes, with sources")

st.write(
    "Upload lecture PDFs, text notes or photos of handwritten/printed "
    "notes. Ask a question and get an answer with the exact source "
    "passages, or generate a quick quiz to test yourself."
)

# ============================================================
# TESSERACT SETUP (only needed for image notes)
# ============================================================

tesseract_path = shutil.which("tesseract")

WINDOWS_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

if not tesseract_path:
    for candidate in WINDOWS_PATHS:
        if os.path.exists(candidate):
            tesseract_path = candidate
            break

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

try:
    pytesseract.get_tesseract_version()
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# ============================================================
# SETTINGS
# ============================================================

CHUNK_WORDS = 120      # words per chunk
CHUNK_OVERLAP = 30     # words shared between neighbouring chunks
MIN_SIMILARITY = 0.08  # below this, we say "not found in notes"
CLAUDE_MODEL = "claude-sonnet-5-5"

# ============================================================
# TEXT EXTRACTION
# ============================================================

def preprocess_for_ocr(image):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    enlarged = cv2.resize(gray, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(enlarged, None, 15, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def ocr_image(image):
    """Return (text, average_confidence) for an image of notes."""

    data = pytesseract.image_to_data(
        preprocess_for_ocr(image),
        config="--oem 3 --psm 6",
        output_type=Output.DICT
    )

    words, confs = [], []
    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = 0
        if word and conf >= 30:
            words.append(word)
            confs.append(conf)

    avg = sum(confs) / len(confs) if confs else 0
    return " ".join(words), avg


def read_file(uploaded):
    """Return a list of pages: {source, page, text, confidence}."""

    name = uploaded.name
    ext = name.lower().rsplit(".", 1)[-1]
    pages = []

    if ext == "pdf":
        reader = PdfReader(uploaded)
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append({"source": name, "page": i,
                          "text": text, "confidence": None})

    elif ext in ("txt", "md"):
        text = uploaded.read().decode("utf-8", errors="ignore")
        pages.append({"source": name, "page": 1,
                      "text": text.strip(), "confidence": None})

    elif ext in ("png", "jpg", "jpeg"):
        if not OCR_AVAILABLE:
            st.warning(f"Skipped {name}: Tesseract OCR is not installed.")
            return []
        image = Image.open(uploaded).convert("RGB")
        text, conf = ocr_image(image)
        pages.append({"source": name, "page": 1,
                      "text": text, "confidence": conf})

    return pages

# ============================================================
# CHUNKING + RETRIEVAL
# ============================================================

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def build_chunks(pages):
    chunks = []
    step = CHUNK_WORDS - CHUNK_OVERLAP

    for p in pages:
        words = clean(p["text"]).split()
        for start in range(0, max(len(words), 1), step):
            piece = words[start:start + CHUNK_WORDS]
            if len(piece) < 15:
                continue
            chunks.append({
                "source": p["source"],
                "page": p["page"],
                "text": " ".join(piece),
            })
    return chunks


@st.cache_resource(show_spinner=False)
def build_index(texts_tuple):
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts_tuple)
    return vectorizer, matrix


def retrieve(question, chunks, vectorizer, matrix, k=4):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).ravel()
    order = scores.argsort()[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in order if scores[i] > 0]

# ============================================================
# ANSWERING
# ============================================================

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extractive_answer(question, hits):
    """No API key? Pick the sentences most similar to the question."""

    sentences = []
    for chunk, _ in hits:
        for s in split_sentences(chunk["text"]):
            if len(s.split()) >= 5:
                sentences.append(s)

    if not sentences:
        return None

    vec = TfidfVectorizer(stop_words="english").fit(sentences + [question])
    scores = cosine_similarity(
        vec.transform([question]), vec.transform(sentences)
    ).ravel()

    best = scores.argsort()[::-1][:3]
    best = sorted(best)  # keep original reading order
    return " ".join(sentences[i] for i in best if scores[i] > 0)


def claude_answer(question, hits):
    """Optional: write a proper answer with Claude, grounded in notes."""

    import anthropic

    context = "\n\n".join(
        f"[{i}] ({c['source']}, p.{c['page']}) {c['text']}"
        for i, (c, _) in enumerate(hits, start=1)
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=(
            "You answer questions using ONLY the numbered note excerpts "
            "provided. Cite excerpts like [1], [2]. If the excerpts do "
            "not contain the answer, say you could not find it in the "
            "notes. Do not use outside knowledge."
        ),
        messages=[{
            "role": "user",
            "content": f"Notes:\n{context}\n\nQuestion: {question}"
        }],
    )
    return response.content[0].text

# ============================================================
# QUIZ GENERATION (fill-in-the-blank)
# ============================================================

def make_quiz(chunks, n_questions=5, seed=None):
    rng = random.Random(seed)

    sentences = []
    for c in chunks:
        for s in split_sentences(c["text"]):
            if 8 <= len(s.split()) <= 35:
                sentences.append((s, c["source"], c["page"]))

    if len(sentences) < 4:
        return []

    vec = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[A-Za-z]{5,}\b",
    )
    try:
        matrix = vec.fit_transform([s[0] for s in sentences])
    except ValueError:
        return []

    vocab = np.array(vec.get_feature_names_out())

    candidates = []
    for i, (sentence, source, page) in enumerate(sentences):
        row = matrix[i].toarray().ravel()
        if row.max() == 0:
            continue
        answer = vocab[row.argmax()]
        candidates.append((sentence, source, page, answer))

    rng.shuffle(candidates)
    all_answers = list({c[3] for c in candidates})
    quiz = []

    for sentence, source, page, answer in candidates:
        if len(quiz) >= n_questions:
            break

        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        if not pattern.search(sentence):
            continue

        blanked = pattern.sub("_____", sentence, count=1)

        pool = [a for a in all_answers if a.lower() != answer.lower()]
        if len(pool) < 3:
            continue

        options = rng.sample(pool, 3) + [answer]
        rng.shuffle(options)

        quiz.append({
            "question": blanked,
            "options": options,
            "answer": answer,
            "source": f"{source}, p.{page}",
        })

    return quiz

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "📤 Upload Notes (PDF, TXT, MD or image)",
    type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ============================================================
# MAIN PROGRAM
# ============================================================

if uploaded_files:

    # --------------------------------------------------------
    # READ NOTES
    # --------------------------------------------------------

    with st.spinner("📖 Reading your notes..."):
        pages = []
        for f in uploaded_files:
            try:
                pages.extend(read_file(f))
            except Exception as error:
                st.error(f"Could not read {f.name}: {error}")

    chunks = build_chunks(pages)

    if not chunks:
        st.warning(
            "No readable text found. If your PDF is scanned, upload "
            "page photos as images instead so OCR can read them."
        )
        st.stop()

    vectorizer, matrix = build_index(tuple(c["text"] for c in chunks))

    # --------------------------------------------------------
    # NOTES OVERVIEW
    # --------------------------------------------------------

    st.divider()
    st.header("🗂️ Notes Overview")

    total_words = sum(len(clean(p["text"]).split()) for p in pages)

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Files", len(uploaded_files))
    o2.metric("Pages", len(pages))
    o3.metric("Words", f"{total_words:,}")
    o4.metric("Searchable chunks", len(chunks))

    overview = pd.DataFrame([
        {
            "File": p["source"],
            "Page": p["page"],
            "Words": len(clean(p["text"]).split()),
            "OCR confidence": (
                f"{p['confidence']:.0f}%" if p["confidence"] is not None
                else "—"
            ),
        }
        for p in pages
    ])
    st.dataframe(overview, use_container_width=True, hide_index=True)

    empty_pages = overview[overview["Words"] < 10]
    if len(empty_pages):
        st.warning(
            f"⚠️ {len(empty_pages)} page(s) have almost no text "
            "(maybe scanned images). Their content can't be searched."
        )

    low_conf = [p for p in pages
                if p["confidence"] is not None and p["confidence"] < 60]
    if low_conf:
        st.warning(
            "📉 Some photos were read with low confidence — answers "
            "from them may be unreliable. Try sharper, well-lit photos."
        )

    # --------------------------------------------------------
    # QUESTION & ANSWER
    # --------------------------------------------------------

    st.divider()
    st.header("💬 Ask Your Notes")

    use_claude = False
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            use_claude = st.toggle(
                "Use Claude to write the answer", value=True
            )
        except ImportError:
            st.caption("Install `anthropic` to enable AI-written answers.")
    else:
        st.caption(
            "Tip: set ANTHROPIC_API_KEY to get written answers. "
            "Without it, the app shows the most relevant sentences."
        )

    question = st.text_input("Your question")

    if question:

        hits = retrieve(question, chunks, vectorizer, matrix)

        if not hits or hits[0][1] < MIN_SIMILARITY:
            st.info(
                "🤷 I couldn't find this in your notes. Try different "
                "keywords, or upload the relevant lecture."
            )
        else:
            with st.spinner("🤖 Finding the answer..."):
                answer = None
                if use_claude:
                    try:
                        answer = claude_answer(question, hits)
                    except Exception as error:
                        st.warning(f"Claude unavailable ({error}). "
                                   "Showing extracted sentences instead.")
                if answer is None:
                    answer = extractive_answer(question, hits)

            st.subheader("✅ Answer")
            st.write(answer or "No clear answer found in the notes.")

            st.subheader("📎 Sources")
            for i, (chunk, score) in enumerate(hits, start=1):
                with st.expander(
                    f"[{i}] {chunk['source']} — page {chunk['page']} "
                    f"(match {score:.0%})"
                ):
                    st.write(chunk["text"])

            if hits[0][1] < 0.2:
                st.warning(
                    "⚠️ Weak match. The answer may only be loosely "
                    "related — check the sources above."
                )

    # --------------------------------------------------------
    # QUIZ
    # --------------------------------------------------------

    st.divider()
    st.header("📝 Quick Quiz")

    n_q = st.slider("Number of questions", 3, 10, 5)

    if st.button("🎲 Generate quiz"):
        st.session_state["quiz"] = make_quiz(
            chunks, n_q, seed=random.randint(0, 10**6)
        )
        st.session_state["checked"] = False

    quiz = st.session_state.get("quiz", [])

    if quiz:
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i + 1}.** {q['question']}")
            st.radio(
                "Pick one",
                q["options"],
                index=None,
                key=f"quiz_{i}",
                label_visibility="collapsed"
            )

        if st.button("✔️ Check answers"):
            st.session_state["checked"] = True

        if st.session_state.get("checked"):
            score = 0
            for i, q in enumerate(quiz):
                picked = st.session_state.get(f"quiz_{i}")
                if picked and picked.lower() == q["answer"].lower():
                    score += 1
                    st.success(f"Q{i + 1}: Correct — {q['answer']}")
                else:
                    st.error(
                        f"Q{i + 1}: Answer was **{q['answer']}** "
                        f"({q['source']})"
                    )
            st.metric("Score", f"{score} / {len(quiz)}")
    elif "quiz" in st.session_state:
        st.info("Not enough text to build a quiz. Upload more notes.")

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()
    st.success("🎯 StudyNotes AI is ready!")

    st.caption(
        "StudyNotes AI splits your notes into chunks, finds the ones "
        "that best match your question with TF-IDF search, and answers "
        "only from those passages. Image notes are read with Tesseract "
        "OCR. Always double-check important facts against your "
        "original notes."
    )