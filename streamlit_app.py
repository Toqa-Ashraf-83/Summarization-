import streamlit as st
from transformers import pipeline

# عنوان التطبيق
st.title("Twitter Sentiment Analysis")
st.write("📊 Type any sentence and its sentiment will be analyzed (positive, negative, neutral)")

# تحميل الموديل من Hugging Face
@st.cache_resource
def load_model():
    model = pipeline("sentiment-analysis", model="bert-base-uncased")
    return model

model = load_model()

# إدخال النص من المستخدم
user_input = st.text_area("✏️ Type text here")

if st.button("Analysis"):
    if user_input.strip() != "":
        result = model(user_input)[0]
        st.write(f"**Classification:** {result['label']}")
        st.write(f"**Ratio:** {result['score']:.2f}")
    else:
        st.warning("⚠️ Please write a text first.ً")