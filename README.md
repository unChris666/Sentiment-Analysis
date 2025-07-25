# Sentiment Analysis of ChatGPT Reviews using Deep Learning

This repository contains a comprehensive analysis and comparison of different deep learning models for classifying the sentiment of user reviews for ChatGPT. The project explores architectures ranging from classic recurrent neural networks (RNNs) to state-of-the-art Transformer models.

---

## 📌 Project Overview

The primary goal of this project is to build and evaluate an effective sentiment analysis model for app reviews. The dataset consists of user reviews for the ChatGPT application, which are categorized into positive, negative, and neutral sentiments.

A key challenge in this dataset is the significant class imbalance, with a majority of reviews being positive. This project implements and compares three distinct modeling schemes, each employing strategies to mitigate the effects of this imbalance.

---

## ⭐ Key Features

- **Data Preprocessing**: A standardized pipeline for cleaning and preparing text data.
- **Automated Labeling**: Use of the VADER sentiment lexicon to programmatically label the dataset.
- **Three Modeling Schemes**:
  - **Bi-GRU with GloVe Embeddings**: A classic RNN approach using pre-trained word embeddings.
  - **BERT Fine-Tuning**: Fine-tuning the `bert-base-uncased` model for sequence classification.
  - **RoBERTa Fine-Tuning**: Fine-tuning the more robust `roberta-base` model.
- **Handling Class Imbalance**: All models utilize class weighting in the loss function to give more importance to minority classes (negative and neutral).

---

## 🔍 Methodology

### 1. Dataset
- **Source**: "ChatGPT Cleaned Reviews" dataset, originally with 99,000 reviews.
- **Columns Used**: `content` (review text) and `vader_sentiment` (generated label).

### 2. Data Preprocessing & Labeling
The text is cleaned and normalized using:
- Lowercasing
- Noise Removal (punctuation, numbers)
- Tokenization
- Stopword Removal (NLTK)
- Lemmatization (spaCy)

**VADER labeling rules**:
- Compound ≥ 0.05 → `positive`
- Compound ≤ -0.05 → `negative`
- Otherwise → `neutral`

**Class Distribution**:
- Positive: 79,455
- Negative: 12,250
- Neutral: 7,295

---

## 🧠 Modeling Architectures

### Scheme I: Bi-GRU with GloVe Embeddings
- **Embeddings**: GloVe 100D, trainable
- **Architecture**:
  - Embedding Layer
  - Bidirectional GRU (128 + 64 units)
  - Dropout(0.5)
  - Dense Softmax output (3-class)
- **Framework**: TensorFlow / Keras

### Scheme II: BERT Fine-Tuning
- **Base**: `bert-base-uncased`
- **Modifications**:
  - Added classification head
  - Fine-tuned on review data
  - Weighted loss with custom `WeightedLossTrainer`
- **Framework**: PyTorch + Hugging Face Transformers

### Scheme III: RoBERTa Fine-Tuning
- **Base**: `roberta-base`
- **Modifications**:
  - Classification head added
  - Weighted CrossEntropy loss in PyTorch
- **Framework**: PyTorch + Hugging Face Transformers

---

## 📈 Results

| Model                     | Test Accuracy | Weighted F1 | Weighted Precision | Weighted Recall |
|--------------------------|---------------|-------------|--------------------|-----------------|
| Bi-GRU + GloVe           | 91.06%        | -           | -                  | -               |
| BERT (bert-base-uncased) | 92.28%        | 0.9252      | 0.9301             | 0.9228          |
| RoBERTa (roberta-base)   | **93.05%**    | **0.93**    | **0.93**           | **0.93**        |

The RoBERTa-based model achieved the best performance across all metrics.

---

## 🧪 Inference Examples (RoBERTa)

| Review                                               | Predicted Sentiment |
|------------------------------------------------------|----------------------|
| "This app is absolutely fantastic! It works perfectly..." | positive         |
| "I'm really disappointed with the latest update..."       | negative         |
| "The application is okay, it does what it says..."        | positive         |
| "I hate this, it's the worst experience I've ever had."   | negative         |
| "just it"                                                 | neutral          |

---

## ⚙️ How to Use

Install dependencies:
```bash
pip install pandas tensorflow torch transformers scikit-learn nltk spacy
```

Download NLTK and spaCy resources:
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Inference with RoBERTa (Scheme III)
```python
import torch
from transformers import AutoTokenizer

# Assume 'model' and 'tokenizer' are loaded and trained
# model = AutoModelForSequenceClassification.from_pretrained(...)
# tokenizer = AutoTokenizer.from_pretrained(...)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

reverse_label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

def predict_sentiment(text, model, tokenizer, device, max_len=128):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_index = torch.argmax(outputs.logits, dim=1).item()
        return reverse_label_map[pred_index]

# Example
new_review = "Wow, I love it! So useful and easy to use."
predicted_sentiment = predict_sentiment(new_review, model, tokenizer, device)
print(f"Review: '{new_review}'\nPredicted Sentiment: {predicted_sentiment}")
```

---

## 👨‍💻 Author

Developed by Nathanael Biran as part of an advanced NLP project.  
For feedback or collaboration, feel free to connect via GitHub.

---
