from pathlib import Path
import html

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------- Page setup --------------------

st.set_page_config(
    page_title="QueryNova AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------- Custom design --------------------

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 50% 0%,
                #1b3154 0%,
                #0b1425 45%,
                #080e1b 100%
            );
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        max-width: 1000px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 55px;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        font-size: 24px;
    }

    .brand-name {
        color: #ffffff;
        font-size: 22px;
        font-weight: 750;
        letter-spacing: 0.4px;
        line-height: 1.3;
    }

    .brand-subtitle {
        color: #9aabc5;
        font-size: 12px;
        margin-top: 3px;
    }

    .hero {
        text-align: center;
        margin: 0 auto 28px auto;
    }

    .hero-tag {
        display: inline-block;
        color: #a9caff;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 100px;
        padding: 7px 14px;
        font-size: 12px;
        letter-spacing: 0.5px;
        margin-bottom: 20px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(32px, 5vw, 48px);
        line-height: 1.15;
        font-weight: 750;
        letter-spacing: -1.4px;
        margin: 0 0 16px 0;
    }

    .hero p {
        color: #aab8cd;
        font-size: 16px;
        line-height: 1.7;
        max-width: 600px;
        margin: 0 auto;
    }

    .question-label {
        color: #e2e8f0;
        font-size: 14px;
        font-weight: 600;
        margin: 0 0 9px 3px;
    }

    [data-testid="stForm"] {
        background: rgba(19, 32, 53, 0.92);
        border: 1px solid #2b3d58;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.18);
    }

    [data-testid="stTextInput"] input {
        background: #f8fafc !important;
        color: #111827 !important;
        border: 1px solid #dbe4f0 !important;
        border-radius: 12px !important;
        min-height: 52px;
        font-size: 15px;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #7b8798 !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.18) !important;
    }

    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 48px;
        color: #ffffff !important;
        background: linear-gradient(90deg, #2563eb, #4f46e5) !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 15px;
        font-weight: 650;
        transition: 0.2s ease;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
        border: none !important;
        transform: translateY(-1px);
    }

    .suggestions {
        text-align: center;
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.8;
        margin-top: 17px;
        margin-bottom: 35px;
    }

    .answer-card {
        background: rgba(17, 29, 49, 0.94);
        border: 1px solid #2b3d58;
        border-radius: 18px;
        padding: 25px 27px;
        margin-top: 25px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.14);
    }

    .answer-heading {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #93c5fd;
        font-size: 13px;
        font-weight: 750;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 18px;
    }

    .answer-icon {
        width: 34px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: rgba(59, 130, 246, 0.16);
        font-size: 17px;
    }

    .answer-text {
        color: #f1f5f9;
        font-size: 16px;
        line-height: 1.85;
        overflow-wrap: anywhere;
    }

    .footer {
        text-align: center;
        color: #71829d;
        font-size: 12px;
        margin-top: 55px;
        padding-top: 20px;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    @media (max-width: 640px) {
        .block-container {
            padding-top: 1.5rem;
        }

        .brand {
            margin-bottom: 38px;
        }

        [data-testid="stForm"] {
            padding: 15px;
        }

        .answer-card {
            padding: 20px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Dataset loading --------------------

DATA_PATH = Path(__file__).parent / "data" / "task1dataset.csv"
SAMPLE_PATH = Path(__file__).parent / "data" / "sample_qa.csv"


@st.cache_data
def load_data(path_string: str) -> pd.DataFrame:
    df = pd.read_csv(path_string)

    # Accept common variations in the column names.
    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    question_column = next(
        (
            columns[name]
            for name in columns
            if name in {"question", "questions", "query"}
        ),
        None,
    )

    answer_column = next(
        (
            columns[name]
            for name in columns
            if name in {"answer", "answers", "response"}
        ),
        None,
    )

    if question_column is None or answer_column is None:
        raise ValueError(
            "The dataset must contain Question and Answer columns."
        )

    clean_data = df[[question_column, answer_column]].copy()
    clean_data.columns = ["Question", "Answer"]

    clean_data = clean_data.dropna()
    clean_data["Question"] = (
        clean_data["Question"].astype(str).str.strip()
    )
    clean_data["Answer"] = (
        clean_data["Answer"].astype(str).str.strip()
    )

    clean_data = clean_data[
        (clean_data["Question"] != "")
        & (clean_data["Answer"] != "")
    ]

    if clean_data.empty:
        raise ValueError("No usable question-answer pairs were found.")

    return clean_data.reset_index(drop=True)


def get_data_path() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    return SAMPLE_PATH


# -------------------- Question answering --------------------

def find_answer(
    user_question: str,
    data: pd.DataFrame,
    threshold: float = 0.22,
):
    questions = data["Question"].tolist()


    # First, check whether the question closely matches a
    # question already present in the dataset.
    normalized_input = " ".join(user_question.lower().split())

    normalized_questions = [
        " ".join(q.lower().split()) for q in questions
    ]

    for index, stored_question in enumerate(normalized_questions):
        if normalized_input == stored_question:
            return data.iloc[index]["Answer"]

    # Otherwise, use TF-IDF similarity.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        questions + [user_question]
    )

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1],
    ).ravel()

    best_index = int(similarities.argmax())
    best_score = float(similarities[best_index])

    # Reject weak matches instead of returning an unrelated answer.
    if best_score < 0.35:
        return None

    return data.iloc[best_index]["Answer"]


# -------------------- Website header --------------------

st.markdown(
    """
    <div class="brand">
        <div class="brand-icon">✦</div>
        <div>
            <div class="brand-name">QueryNova AI</div>
            <div class="brand-subtitle">Your knowledge assistant</div>
        </div>
    </div>

    <div class="hero">
        <div class="hero-tag">✧ &nbsp; ASK. EXPLORE. DISCOVER.</div>
        <h1>What would you like<br>to know?</h1>
        
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------- Load dataset --------------------

try:
    data_path = get_data_path()
    data = load_data(str(data_path))
except Exception:
    st.error(
        "We couldn't load the question-answer service. "
        "Please check the dataset file and try again."
    )
    st.stop()


# -------------------- Question input --------------------

st.markdown(
    '<div class="question-label">Your question</div>',
    unsafe_allow_html=True,
)

with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Enter your question",
        placeholder="e.g. What does SPIT stand for?",
        label_visibility="collapsed",
    )

    submitted = st.form_submit_button("✦  Find my answer")


# -------------------- Display answer --------------------

if submitted:
    if not question.strip():
        st.warning("Please type a question before continuing.")
    else:
        answer = find_answer(question.strip(), data)

        if answer is None:
            st.warning(
                "I couldn't find a suitable answer to that question. "
                "Try asking it another way or asking about a related topic."
            )
        else:
            safe_answer = html.escape(str(answer))

            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-heading">
                        <span class="answer-icon">✦</span>
                        Your answer
                    </div>
                    <div class="answer-text">{safe_answer}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------- Example prompts and footer --------------------

st.markdown(
    """
    <div class="suggestions">
        Not sure where to start?<br>
        Try: <b>What is SPIT?</b> &nbsp; · &nbsp;
        <b>What does SPIT stand for?</b>
    </div>

    <div class="footer">
        SWYNEX AI &nbsp; · &nbsp; Your questions, answered.
    </div>
    """,
    unsafe_allow_html=True,
)

