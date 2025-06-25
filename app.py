import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.title("📰 Fake News Detector")
st.markdown("Paste a news article below and check if it's **Fake** or **Real**.")

user_input = st.text_area("📝 News Article Text", height=200)

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]
        label = "✅ Real News" if pred == 1 else "🚫 Fake News"
        st.success(f"**Result:** {label}")
