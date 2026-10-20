import html
import os
import re
import time
from pathlib import Path

# Reduce TensorFlow startup messages where possible.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence


# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "sentiment_lstm.keras"
VOCAB_PATH = MODELS_DIR / "imdb_word_index.npz"

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 200
PADDING_TYPE = "post"
TRUNCATING_TYPE = "post"
SENTIMENT_THRESHOLD = 0.5


# ============================================================
# STREAMLIT PAGE SETTINGS
# ============================================================
st.set_page_config(
    page_title="SentimentAI — Movie Review Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM DARK THEME
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #07090e;
        background-image:
            radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.08) 0px, transparent 50%);
        color: #f1f5f9;
    }

    section[data-testid="stSidebar"] {
        background-color: #0c101a !important;
        border-right: 1px solid rgba(139, 92, 246, 0.18) !important;
    }

    .hero-container {
        text-align: center;
        padding: 2.2rem 1rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.85), rgba(11, 15, 25, 0.96));
        border: 1px solid rgba(139, 92, 246, 0.25);
        box-shadow: 0 10px 30px -10px rgba(124, 58, 237, 0.2);
        margin-bottom: 1.8rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 9999px;
        color: #c4b5fd;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(1.8rem, 4vw, 2.7rem);
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 40%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.75rem;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 650px;
        margin: 0 auto;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 14px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .result-card-pos, .result-card-neg {
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .result-card-pos {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.12), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .result-card-neg {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.12), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .sentiment-tag-pos, .sentiment-tag-neg {
        display: inline-block;
        font-size: 1.15rem;
        font-weight: 700;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        margin-top: 0.4rem;
    }

    .sentiment-tag-pos {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .sentiment-tag-neg {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .step-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
        height: 100%;
    }

    .step-num {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        color: #a78bfa;
        background: rgba(139, 92, 246, 0.2);
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }

    .step-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.35rem;
    }

    .step-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    .history-card {
        background: rgba(15, 23, 42, 0.5);
        border-left: 3px solid #8b5cf6;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #7c3aed, #6366f1);
        color: white;
        border: none;
    }

    textarea {
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 10px !important;
    }

    .muted {
        color: #94a3b8;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "review_input" not in st.session_state:
    st.session_state.review_input = ""


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner="Loading trained LSTM model and IMDb vocabulary...")
def load_model_and_vocabulary(model_path: str, vocab_path: str):
    """Load the saved Keras model and the word-index file produced by train.py."""
    model = tf.keras.models.load_model(model_path)

    with np.load(vocab_path, allow_pickle=False) as data:
        words = data["words"].tolist()
        indices = data["indices"].tolist()

    word_index = dict(zip(words, indices))
    return model, word_index


def encode_review(review_text: str, word_index: dict, max_len: int = MAX_SEQUENCE_LENGTH):
    """
    Encode text using the IMDb Keras dataset convention:
    1 = start token, 2 = out-of-vocabulary token, normal word indices + 3.
    IDs >= VOCAB_SIZE are replaced by 2, matching imdb.load_data(num_words=10000).
    """
    # Keras text tokenization lowercases and splits punctuation similarly to IMDb.
    words = text_to_word_sequence(review_text)
    sequence = [1]  # IMDb start token

    for word in words:
        original_index = word_index.get(word)

        if original_index is None:
            sequence.append(2)
            continue

        imdb_index = int(original_index) + 3

        if imdb_index >= VOCAB_SIZE:
            sequence.append(2)
        else:
            sequence.append(imdb_index)

    padded = pad_sequences(
        [sequence],
        maxlen=max_len,
        padding=PADDING_TYPE,
        truncating=TRUNCATING_TYPE,
        value=0,
    )
    return padded, len(words)


def predict_sentiment(review_text: str, model, word_index: dict):
    """Return a prediction from the actual trained model."""
    started = time.perf_counter()
    padded_review, token_count = encode_review(review_text, word_index)

    prediction = model.predict(padded_review, verbose=0)
    positive_score = float(np.asarray(prediction).reshape(-1)[0])

    is_positive = positive_score >= SENTIMENT_THRESHOLD
    label = "Positive" if is_positive else "Negative"
    class_score = positive_score if is_positive else 1.0 - positive_score

    return {
        "label": label,
        "score": positive_score,
        "confidence": class_score * 100.0,
        "is_positive": is_positive,
        "inference_time_ms": (time.perf_counter() - started) * 1000.0,
        "token_count": token_count,
    }


# ============================================================
# SAMPLE REVIEWS
# ============================================================
SAMPLE_REVIEWS = [
    {
        "type": "positive",
        "title": "A Great Film",
        "icon": "🟢",
        "text": (
            "An absolutely wonderful film with brilliant acting, a moving story, "
            "and beautiful direction. I enjoyed every minute and would happily watch it again."
        ),
    },
    {
        "type": "positive",
        "title": "Stellar Acting",
        "icon": "🟢",
        "text": (
            "The performances were excellent and the story kept me interested throughout. "
            "A memorable movie with a satisfying ending."
        ),
    },
    {
        "type": "negative",
        "title": "A Disappointment",
        "icon": "🔴",
        "text": (
            "A complete waste of time. The story was boring, the acting felt flat, "
            "and the ending was disappointing. I would not recommend this movie."
        ),
    },
    {
        "type": "negative",
        "title": "Poorly Made",
        "icon": "🔴",
        "text": (
            "Terrible dialogue, weak characters, and painfully slow pacing made this film "
            "a frustrating experience. I regretted watching it."
        ),
    },
]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎬 SentimentAI")
    st.caption("Deep Learning Movie Review Analyzer")
    st.markdown("---")
    st.markdown("### 📦 Model Engine Status")

    model_exists = MODEL_PATH.is_file()
    vocab_exists = VOCAB_PATH.is_file()

    if model_exists and vocab_exists:
        st.success("Model and IMDb vocabulary found")
        st.caption(f"Model: `{MODEL_PATH.name}`")
        st.caption(f"Vocabulary: `{VOCAB_PATH.name}`")
    else:
        st.warning("Model files not found")
        st.markdown(
            "Expected files inside the `models` folder:\n"
            "- `sentiment_lstm.keras`\n"
            "- `imdb_word_index.npz`"
        )

    with st.expander("⚙️ Pipeline specifications"):
        st.markdown(
            f"""
            - **Architecture:** Embedding → LSTM → Dense
            - **Vocabulary size:** {VOCAB_SIZE:,}
            - **Maximum sequence length:** {MAX_SEQUENCE_LENGTH}
            - **Padding:** `{PADDING_TYPE}`
            - **Truncation:** `{TRUNCATING_TYPE}`
            - **Decision threshold:** {SENTIMENT_THRESHOLD}
            """
        )

    st.markdown("---")
    st.markdown("### 🕒 Session History")

    if not st.session_state.history:
        st.caption("Your predictions will appear here during this session.")
    else:
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        for item in reversed(st.session_state.history[-6:]):
            color = "#34d399" if item["label"] == "Positive" else "#f87171"
            safe_snippet = html.escape(item["snippet"])
            st.markdown(
                f"""
                <div class="history-card">
                    <div style="display:flex;justify-content:space-between;gap:8px;">
                        <span style="color:{color};font-weight:700;">{item['label']}</span>
                        <span style="color:#94a3b8;">{item['confidence']:.1f}%</span>
                    </div>
                    <div style="color:#cbd5e1;font-size:0.78rem;">
                        {safe_snippet}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("SentimentAI • TensorFlow + Streamlit")


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">✨ Deep Learning • Natural Language Processing</div>
        <div class="hero-title">SentimentAI — Movie Review Analyzer</div>
        <div class="hero-subtitle">Understand the emotion behind every review.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL IF FILES EXIST
# ============================================================
model = None
word_index = None
load_error = None

if model_exists and vocab_exists:
    try:
        model, word_index = load_model_and_vocabulary(
            str(MODEL_PATH), str(VOCAB_PATH)
        )
    except Exception as exc:
        load_error = str(exc)
        st.error(f"Could not load the model files: {load_error}")
else:
    st.warning(
        "The trained model or vocabulary file is missing. "
        "Check that both files are inside the project's `models` folder."
    )


# ============================================================
# REVIEW ANALYSIS UI
# ============================================================
st.markdown("### 💡 Try an example review")
st.caption("Choose an example to load it into the review box.")

sample_columns = st.columns(len(SAMPLE_REVIEWS))
for index, sample in enumerate(SAMPLE_REVIEWS):
    with sample_columns[index]:
        if st.button(
            f"{sample['icon']} {sample['title']}",
            key=f"sample_{index}",
            use_container_width=True,
        ):
            st.session_state.review_input = sample["text"]
            st.rerun()

st.markdown("### ✍️ Enter a movie review")

user_review = st.text_area(
    "Movie review",
    key="review_input",
    height=160,
    placeholder=(
        "Type or paste a movie review here...\n\n"
        "Example: The story was engaging and the performances were excellent."
    ),
)

word_count = len(user_review.split()) if user_review.strip() else 0
character_count = len(user_review)
stat_col1, stat_col2, stat_col3 = st.columns([1, 1, 2])
stat_col1.metric("Words", word_count)
stat_col2.metric("Characters", character_count)
with stat_col3:
    st.caption("The model processes up to 200 tokens after IMDb vocabulary encoding.")

action_col, clear_col, _ = st.columns([1.4, 0.8, 2.2])
with action_col:
    analyze_clicked = st.button(
        "⚡ Analyze review",
        type="primary",
        use_container_width=True,
        disabled=(model is None or word_index is None),
    )
with clear_col:
    if st.button("Clear", use_container_width=True):
        st.session_state.review_input = ""
        st.rerun()


# ============================================================
# PREDICTION RESULT
# ============================================================
if analyze_clicked:
    review = user_review.strip()

    if not review:
        st.warning("Please enter a movie review before analyzing.")
    else:
        try:
            with st.spinner("Encoding words and running the trained LSTM model..."):
                result = predict_sentiment(review, model, word_index)

            # Keep recent predictions only in this Streamlit session.
            st.session_state.history.append(
                {
                    "snippet": review[:65] + ("..." if len(review) > 65 else ""),
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "score": result["score"],
                }
            )
            st.session_state.history = st.session_state.history[-20:]

            is_positive = result["is_positive"]
            card_class = "result-card-pos" if is_positive else "result-card-neg"
            tag_class = "sentiment-tag-pos" if is_positive else "sentiment-tag-neg"
            sentiment_text = "🟢 Positive Sentiment" if is_positive else "🔴 Negative Sentiment"

            positive_percent = result["score"] * 100.0
            negative_percent = (1.0 - result["score"]) * 100.0

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;">
                        <div>
                            <div style="font-size:0.85rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">
                                Predicted sentiment
                            </div>
                            <div class="{tag_class}">{sentiment_text}</div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:0.85rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.05em;">
                                Model score for predicted class
                            </div>
                            <div style="font-size:2rem;font-weight:800;color:#f8fafc;">
                                {result['confidence']:.1f}%
                            </div>
                            <div style="font-size:0.75rem;color:#94a3b8;">
                                Raw sigmoid output: {result['score']:.4f}
                            </div>
                        </div>
                    </div>
                    <div style="margin-top:1.2rem;">
                        <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;font-size:0.8rem;margin-bottom:0.35rem;">
                            <span style="color:#f87171;">Negative score: {negative_percent:.1f}%</span>
                            <span style="color:#34d399;">Positive score: {positive_percent:.1f}%</span>
                        </div>
                        <div style="width:100%;height:14px;background:#1e293b;border-radius:999px;overflow:hidden;border:1px solid rgba(139,92,246,0.25);">
                            <div style="width:{positive_percent:.2f}%;height:100%;background:linear-gradient(90deg,#ef4444,#8b5cf6,#10b981);"></div>
                        </div>
                    </div>
                    <div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:0.8rem;font-size:0.75rem;color:#94a3b8;">
                        <span>Words tokenized: {result['token_count']}</span>
                        <span>Inference time: {result['inference_time_ms']:.1f} ms</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Note: the displayed score comes directly from the model and is not "
                "a guarantee that the prediction is correct or a calibrated confidence."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the model was trained with vocabulary size 10,000, "
                "sequence length 200, and the saved IMDb word-index file."
            )


# ============================================================
# HOW IT WORKS
# ============================================================
st.markdown("---")
st.markdown("## 🧠 How it works")
st.markdown(
    '<p class="muted">Five stages turn raw review text into a sentiment prediction.</p>',
    unsafe_allow_html=True,
)

steps = [
    (
        "STEP 1",
        "🔤 Tokenization",
        "The review is split into words and mapped to integer IDs from the IMDb vocabulary.",
    ),
    (
        "STEP 2",
        "📏 Padding",
        "The sequence is padded or truncated to 200 positions so it matches the model input.",
    ),
    (
        "STEP 3",
        "🌐 Embedding",
        "The embedding layer maps integer word IDs to trainable numeric vectors.",
    ),
    (
        "STEP 4",
        "🔄 LSTM",
        "The LSTM processes the sequence and learns patterns in the order of words.",
    ),
    (
        "STEP 5",
        "🎯 Prediction",
        "A sigmoid output produces a score between 0 and 1, interpreted using a 0.5 threshold.",
    ),
]

step_columns = st.columns(5)
for column, (number, title, description) in zip(step_columns, steps):
    with column:
        st.markdown(
            f"""
            <div class="step-card">
                <span class="step-num">{number}</span>
                <div class="step-title">{title}</div>
                <div class="step-desc">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")
with st.expander("About this model"):
    st.write(
        "This application uses a trained LSTM neural network built with TensorFlow/Keras "
        "and trained on the IMDb movie-review sentiment dataset. The model predicts "
        "positive or negative sentiment. It may make mistakes, especially with sarcasm, "
        "mixed opinions, unusual wording, or text unlike the training data."
    )
