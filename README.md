#Hate Speech Detection and Detoxification

This project builds an NLP pipeline that detects hate speech and rewrites toxic sentences into neutral ones.
It combines deep learning (BiLSTM + Attention) for classification and a Transformer (T5) model for text detoxification.

###Features

Hate speech detection using BiLSTM + Attention
Word embeddings with FastText
SHAP explanations for model interpretability
Sentence detoxification using T5 Transformer
Simple Flask web interface for real-time prediction
Baseline model using Logistic Regression

###Workflow

Input Sentence
→ Hate Speech Detection (LSTM Model)
→ If toxic → Sentence rewritten by T5
→ Output shown to the user

###Datasets
HateSpeechDatasetBalanced.csv
s-nlp/ParaDetox
textdetox/multilingual_paradetox

###Evaluation

####Classification performance is evaluated using:
Accuracy
Precision / Recall / F1-score
Confusion Matrix

####Text generation quality is evaluated using:
ROUGE score
