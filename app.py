import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from transformers import AutoTokenizer, BertForSequenceClassification
from safetensors.torch import load_file  # Ensure safetensors is installed

# Correct Paths


MODEL_PATH = r"C:\Users\hp\Desktop\Financial News Sentiment Analyzer\model.safetensors"

try:
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", torch_dtype=torch.float16)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", str(e))

TOKENIZER_PATH = r"C:\Users\hp\Desktop\Financial News Sentiment Analyzer\finbert_model"  # Use directory

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print("❌ Error loading tokenizer:", str(e))
    tokenizer = None

# Load model
try:
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")  # Load base FinBERT model
    state_dict = load_file(MODEL_PATH)  # Load safetensors weights
    model.load_state_dict(state_dict)  # Apply weights to model
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None


TOKENIZER_PATH = r"C:\Users\hp\Desktop\Financial News Sentiment Analyzer\finbert_model"
print("Tokenizer folder contains:", os.listdir(TOKENIZER_PATH))

# Sentiment labels
sentiment_labels = ["Negative", "Neutral", "Positive"]

# Function to predict sentiment and confidence scores
def predict_sentiment(text):
    if tokenizer is None or model is None:
        return "Error", 0.0  # Return an error if model or tokenizer failed to load
    
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**encoded_input)
    scores = F.softmax(output.logits, dim=1)
    predicted_class = torch.argmax(scores).item()
    confidence = scores[0][predicted_class].item() * 100  # Convert to percentage
    return sentiment_labels[predicted_class], confidence

# Streamlit UI
st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="wide")
st.title("📈 Financial News Sentiment Analyzer")
st.write("Analyze sentiment of financial news sentences.")

# Sidebar Options
st.sidebar.header("Options")
view_mode = st.sidebar.radio("Choose Mode:", ["Single Prediction", "Batch Analysis"])

# Single Text Prediction
if view_mode == "Single Prediction":
    user_input = st.text_area("Enter a financial news sentence:", "")

    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment, confidence = predict_sentiment(user_input)
            st.markdown(f"**Predicted Sentiment:** {sentiment}")
            st.markdown(f"**Confidence Score:** {confidence:.2f}%")
        else:
            st.warning("Please enter a news sentence.")

# Batch Processing (Upload CSV)
elif view_mode == "Batch Analysis":
    uploaded_file = st.file_uploader("Upload a CSV file with a 'sentence' column", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "sentence" not in df.columns:
            st.error("CSV must contain a column named 'sentence'.")
        else:
            st.write("Processing sentences...")

            # Apply sentiment analysis to all sentences
            df["Sentiment"], df["Confidence"] = zip(*df["sentence"].apply(predict_sentiment))

            # Show results
            st.write(df)

            # Plot Sentiment Distribution
            st.subheader("Sentiment Distribution")
            plt.figure(figsize=(6, 4))
            sns.countplot(x=df["Sentiment"], palette="coolwarm")
            plt.xlabel("Sentiment")
            plt.ylabel("Count")
            st.pyplot(plt)
