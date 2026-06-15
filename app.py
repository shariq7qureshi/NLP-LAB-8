import os
import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Grammar & Writing Assistant",
    page_icon="✍️",
    layout="wide"
)

# -----------------------------
# CSS styling
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
# MODEL LOADING (Explicit TensorFlow Engine)
# -----------------------------
@st.cache_resource
def load_model():
    # Explicitly load the tokenizer and model without using the buggy pipeline wrapper
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("✍️ AI Grammar & Writing Assistant")
st.caption(
    "Powered by a pretrained Hugging Face model • Runs locally via TensorFlow • No API Keys Required"
)

# -----------------------------
# TASK SELECTION
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
# TEXT INPUT
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
    "Fix Grammar": "Correct the grammar and spelling of this text: ",
    "Professional Rewrite": "Rewrite the following text to sound highly professional and formal: ",
    "Simplify Text": "Simplify this text so it is easy to understand: ",
    "Summarize": "Summarize the following text in a few sentences: ",
    "Convert to Email": "Write a professional email based on the following instructions: "
}

# -----------------------------
# EXECUTION & INFERENCE
# -----------------------------
if st.button("✨ Generate"):
    if len(user_text.strip()) == 0:
        st.warning("Please enter some text.")
    else:
        # Construct explicit continuous string instruction
        prompt = prompt_map[mode] + user_text.strip()
        
        with st.spinner("AI is processing your request..."):
            # Step 1: Explicitly encode input text to native TensorFlow tensors
            inputs = tokenizer(prompt, return_tensors="tf")
            
            # Step 2: Manually trigger the TensorFlow model generation loop
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=300,
                early_stopping=True
            )
            
            # Step 3: Decode token arrays back into clean human-readable text
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Result")
        st.success(output)

# -----------------------------
# FOOTER
# -----------------------------
st.divider()
st.info(
"""
Features Matrix:
✅ Grammar Correction • ✅ Professional Rewriting • ✅ Simplification • ✅ Summarization • ✅ Email Conversion
"""
)
