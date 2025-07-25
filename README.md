# Sentiment Analysis of ChatGPT Reviews

This project performs sentiment analysis on user reviews of ChatGPT collected from the Google Play Store. It was developed as part of a machine learning assignment using Python and NLP techniques.

## 📌 Project Objective

The main objective is to classify user reviews into sentiment categories — **positive**, **neutral**, and **negative** — using natural language processing and machine learning techniques. This analysis helps understand public perception and feedback about the ChatGPT application.

## 🛠️ Methodology

1. **Text Preprocessing**:
   - Tokenization with NLTK
   - Stopword removal (NLTK and spaCy)
   - Regular expression cleaning
   - Lowercasing and punctuation removal

2. **Exploratory Data Analysis (EDA)**:
   - Class distribution visualization
   - Word frequency analysis for each sentiment

3. **Feature Engineering**:
   - TF-IDF Vectorization

4. **Modeling**:
   - Models used:
     - Multinomial Naive Bayes
     - Logistic Regression
     - Support Vector Machine (SVM)
   - Evaluation using:
     - Accuracy
     - Classification Report (Precision, Recall, F1-Score)
     - Confusion Matrix

## 📊 Results Summary

- **Best model**: Support Vector Machine (SVM)
- **Accuracy**: ~88% on test data
- **Observations**:
  - Positive reviews dominate the dataset.
  - The classifier performs better on positive and negative classes compared to neutral.

## 📂 Dataset

The dataset used is a cleaned CSV file containing ChatGPT reviews, originally sourced from the Google Play Store.

File path in the notebook:
```
/kaggle/input/sentiment-analysist/chatGPT_clean_reviews.csv
```

## 🧰 Dependencies

Key libraries used:
- `pandas`
- `nltk`
- `spacy`
- `sklearn`
- `matplotlib`, `seaborn` (for EDA)

To install required packages:
```bash
pip install nltk spacy scikit-learn matplotlib seaborn
```

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/sentiment-analysis-chatgpt.git
cd sentiment-analysis-chatgpt
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

3. Run the notebook:
Open `sentimen-analysist-dicoding.ipynb` in Jupyter Notebook or VS Code and run all cells.

## 📌 Author

Developed by Nathanael Biran as part of a Dicoding NLP learning project.  
For collaboration or questions, feel free to reach out via GitHub.

---
