import streamlit as st
from transformers import pipeline

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="T5 Text Summarizer",
    page_icon="📝",
    layout="wide"
)

# ---------------------------------------
# LOAD MODEL
# ---------------------------------------

@st.cache_resource
def load_model():
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer="t5-small"
    )
    return summarizer

summarizer = load_model()

# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("T5 Text Summarization System")

st.write(
    "Enter a paragraph or article below and the T5 model "
    "will generate a concise summary."
)

st.divider()

# ---------------------------------------
# INPUT
# ---------------------------------------

st.subheader("Enter Text")

text = st.text_area(
    "Input Text",
    height=250,
    placeholder="Enter your text here..."
)

# ---------------------------------------
# SETTINGS
# ---------------------------------------

col1, col2 = st.columns(2)

with col1:
    min_length = st.slider(
        "Minimum Summary Length",
        10, 50, 20
    )

with col2:
    max_length = st.slider(
        "Maximum Summary Length",
        30, 150, 80
    )

# ---------------------------------------
# SUMMARIZE BUTTON
# ---------------------------------------

if st.button("Generate Summary"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    elif len(text.split()) < 30:
        st.warning("Please enter at least 30 words.")

    else:
        with st.spinner("Generating summary..."):

            input_text = "summarize: " + text

            result = summarizer(
                input_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            summary = result[0]["summary_text"]

        # --------------------------------
        # OUTPUT
        # --------------------------------

        st.success("Summary Generated Successfully!")

        st.subheader("Generated Summary")

        st.write(summary)

        st.divider()

        # --------------------------------
        # STATISTICS
        # --------------------------------

        original_words = len(text.split())
        summary_words = len(summary.split())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Original Words",
                original_words
            )

        with col2:
            st.metric(
                "Summary Words",
                summary_words
            )

        with col3:
            reduction = (
                (original_words - summary_words)
                / original_words
            ) * 100

            st.metric(
                "Text Reduction",
                f"{reduction:.1f}%"
                  )
