import streamlit as st
from transformers import pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Grammar & Writing Assistant",
    page_icon="✍️",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------

st.markdown("""
<style>

.stApp{
    background-color:#0f172a;
}

h1,h2,h3,h4,h5,p,label{
    color:white;
}

textarea{
    background:white !important;
    color:black !important;
}

.stButton>button{
    background:#10b981;
    color:white;
    border-radius:12px;
    border:none;
    height:3em;
    width:100%;
    font-size:18px;
    font-weight:bold;
}

div[data-testid="stSelectbox"]{
    color:black;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODEL
# -----------------------------

@st.cache_resource
def load_model():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="tf" 
    )
    return pipe

generator = load_model()

# -----------------------------
# HEADER
# -----------------------------

st.title("✍️ AI Grammar & Writing Assistant")

st.caption(
    "Powered by a pretrained Hugging Face model • Runs locally • No API"
)

# -----------------------------
# MODE
# -----------------------------

mode = st.selectbox(

    "Choose Task",

    [

        "Fix Grammar",

        "Professional Rewrite",

        "Simplify Text",

        "Summarize",

        "Convert to Email"

    ]

)

# -----------------------------
# INPUT
# -----------------------------

user_text = st.text_area(

    "Enter your text",

    height=250,

    placeholder="Paste your paragraph here..."

)

# -----------------------------
# PROMPTS
# -----------------------------

prompt_map = {

    "Fix Grammar":

        "Correct the grammar of the following text:\n\n",

    "Professional Rewrite":

        "Rewrite professionally:\n\n",

    "Simplify Text":

        "Simplify the following text:\n\n",

    "Summarize":

        "Summarize:\n\n",

    "Convert to Email":

        "Convert the following into a professional email:\n\n"

}

# -----------------------------
# BUTTON
# -----------------------------

if st.button("✨ Generate"):

    if len(user_text.strip()) == 0:

        st.warning("Please enter some text.")

    else:

        prompt = prompt_map[mode] + user_text

        with st.spinner("AI is writing..."):

            result = generator(

                prompt,

                max_new_tokens=300,

                do_sample=False

            )

        output = result[0]["generated_text"]

        st.subheader("Result")

        st.success(output)

# -----------------------------
# FOOTER
# -----------------------------

st.divider()

st.info(

"""
Features:

✅ Grammar Correction

✅ Professional Rewriting

✅ Simplification

✅ Summarization

✅ Email Conversion

Built using a pretrained Hugging Face model running locally.

"""

)
