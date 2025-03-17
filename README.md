**Report: Financial News Sentiment Analysis Using FinBERT**
**Objective**
The goal of this project is to develop a Financial News Sentiment Analyzer using FinBERT, a transformer-based model specialized in financial text analysis. The system preprocesses financial news, trains a logistic regression classifier, and fine-tunes FinBERT to predict sentiment as Negative, Neutral, or Positive.

**1. Data Preprocessing & Exploration**
Dataset: Loaded from a CSV file (data.csv).
Cleaning Steps:
Removed missing values.
Tokenization and stopword removal using NLTK.
Lemmatization applied for better text representation.
Exploratory Data Analysis (EDA):
Sentiment Distribution visualized using Seaborn.
Word Frequency Analysis conducted using Counter and WordCloud.
Sentence Length Distribution plotted.
**2. Machine Learning Model (Baseline)**
Feature Extraction: TF-IDF vectorization with 5000 features.
Sentiment Classification: Trained a Logistic Regression model.
Performance Evaluation:
Achieved accuracy and classification report metrics.
The model serves as a baseline before fine-tuning FinBERT.
**3. Transformer-Based Model (FinBERT)**
FinBERT Model & Tokenizer:
Used ProsusAI/finbert from Hugging Face.
Tokenized financial news headlines with max length 128.
Dataset Preparation:
Converted text to PyTorch tensors.
Labeled sentiments as integers (Negative = 0, Neutral = 1, Positive = 2).
Split into train (80%) and validation (20%) datasets.
Training Setup:
Fine-tuned FinBERT for 3 epochs.
Used AdamW optimizer and CrossEntropy loss function.
Implemented learning rate scheduler for optimization.
Trained using PyTorch DataLoader.
Evaluation Metrics:
Validation accuracy calculated after each epoch.
**4. Deployment & Prediction**
Fine-Tuned Model Saved: finbert_model
Tokenizer Saved: finbert_tokenizer
Prediction Function Implemented:
Takes input financial news.
Tokenizes and processes with FinBERT.
Predicts sentiment using softmax classification.
Example Prediction:
Input: "The stock market is performing exceptionally well today!"
Predicted Sentiment: Positive
**5. Challenges & Solutions**
Issue	Solution
**Tokenizer path issues**	Verified tokenizer directory and files (config.json, vocab.txt).
**Memory error (os error 1455)**	Increased virtual memory, optimized model loading.
**Mismatch in sentiment labels**	Standardized encoding with integer mapping.
