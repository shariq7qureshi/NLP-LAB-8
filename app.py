
import streamlit as st
from transformers import TFT5ForConditionalGeneration, AutoTokenizer
import tensorflow as tf

MODEL_NAME = "google/flan-t5-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = TFT5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        from_pt=True
    )

    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="MediGuide AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp {background:#0d1117;color:white;}
.block-container {max-width:1000px;}
div[data-testid="stTextArea"] textarea{
background:#f8fff8;color:#111;border:2px solid #2e8b57;border-radius:12px;
}
.stButton>button{
background:#2e8b57;color:white;border-radius:10px;border:none;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 MediGuide AI")
st.caption("Educational health assistant. Not a diagnostic tool.")

symptoms = st.text_area(
    "Describe your symptoms",
    placeholder="Example: fever, cough, sore throat, headache for 3 days..."
)

if st.button("Analyze") and symptoms.strip():
    prompt = f"""
You are a medical education assistant.

Given symptoms:
{symptoms}

Return:
1. Possible conditions (educational only)
2. Brief explanation
3. Questions a doctor may ask
4. General self-care suggestions
5. End with:
'Consult a licensed healthcare professional for diagnosis.'
"""
    ids = tokenizer(prompt, return_tensors="tf").input_ids
    out = model.generate(ids, max_new_tokens=256)
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    st.subheader("Educational Guidance")
    st.write(text)

st.divider()

st.subheader("Health Chat")

q = st.text_input("Ask a health-related question")

if st.button("Ask") and q.strip():
    prompt = f"""
You are a cautious medical education chatbot.
Never diagnose.
Never prescribe medications.
Encourage consulting doctors.

Question:
{q}
"""
    ids = tokenizer(prompt, return_tensors="tf").input_ids
    out = model.generate(ids, max_new_tokens=256)
    ans = tokenizer.decode(out[0], skip_special_tokens=True)
    st.write(ans)

st.warning("""
This application provides educational information only.
It does not diagnose disease, replace medical professionals,
or prescribe treatment. Seek urgent medical care for emergencies.
""")
