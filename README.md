# Fake News Detection System

An advanced machine learning-based system developed for detecting fake news articles using natural language processing and ensemble learning methods. It features a Flask REST API backend and a React frontend for real-time predictions.

---

## Project Overview

This system achieves an accuracy of 96.8% in classifying fake news using a powerful ensemble of Logistic Regression, Random Forest, and XGBoost models. The detection is based on thorough analysis of text content, structural patterns, and linguistic cues. It is designed for scalable, real-time prediction through a modern web interface.

---

## Key Features

* **High Accuracy:** Achieves 96.8% accuracy with an AUC-ROC score of 0.9953  
* **Real-Time Predictions:** Users receive instant classification results via a responsive React interface   
* **Ensemble Learning:** Utilizes a Voting Classifier with Logistic Regression, Random Forest, and XGBoost  
* **Comprehensive Feature Engineering:** Includes TF-IDF, sentiment analysis, keyword detection, and more  
* **REST API Backend:** Flask-based backend enabling smooth and scalable deployment  
* **Modern Frontend:** Built with React for a seamless user experience  
* **Robust Validation:** Trained and tested on 63,121 samples with rigorous cross-validation  

---

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 96.82% |
| Precision | 96.05% |
| Recall    | 96.89% |
| F1-Score  | 96.47% |
| AUC-ROC   | 0.9953 |

---

## Dataset Statistics

* **Total Samples:** 63,121  
* **Training Set:** 50,496 (80%)  
* **Test Set:** 12,625 (20%)  
* **Features:** 8,010 (8,000 TF-IDF + 10 engineered features)  

---

## Data Preprocessing

* Text cleaning and normalization  
* Special character and punctuation handling  
* URL and number tokenization  
* Stop word removal  

---

## Feature Engineering

* **TF-IDF Vectorization:** 8,000 features with 1-2 n-grams  
* **Additional Numerical Features:**
  * Text length and word count  
  * Punctuation usage  
  * Sentiment score (VADER)  
  * Fake news keyword frequency  
  * Capitalization ratio  

---

## Tech Stack

| Layer       | Technology Used                     |
|-------------|--------------------------------------|
| Frontend    | React.js                             |
| Backend     | Flask (Python)                       |
| Model Dev   | Google Colab / Jupyter Notebook      |
| ML Models   | Logistic Regression, Random Forest, XGBoost |
| Libraries   | Scikit-learn, XGBoost, Pandas, NLTK, VADER |
| API Comm    | REST API via Flask                   |


## Model Architecture

* **Ensemble Approach:** Voting Classifier  
* **Base Models:**
  * Logistic Regression (C=1.0, balanced)  
  * Random Forest (100 estimators, max_depth=15)  
  * XGBoost (100 estimators, max_depth=6)  

---

## Validation Strategy

* 3-fold cross-validation  
* Stratified train-test split  
* Detailed evaluation on holdout test set  

---

## Fake News Indicators

* **Sensational Language:** Excess punctuation and capitalization  
* **Emotional Manipulation:** Extreme sentiment values  
* **Clickbait Phrases:** Words like "BREAKING", "SHOCKING", "EXPOSED"  
* **Structural Oddities:** Unusual formatting and length  
* **Linguistic Cues:** Frequency of exclamation marks and questions  

---

## Preprocessing Pipeline

1. Lowercase text normalization  
2. Handling special characters and URLs  
3. Tokenization with stop word filtering  
4. Extraction of structural and linguistic features  
5. Normalization of numerical inputs  

---

## Prediction Confidence Levels

* **High Confidence ( > 0.8 ):** Strong prediction  
* **Medium Confidence ( 0.6 - 0.8 ):** Reliable but moderate certainty  
* **Low Confidence ( < 0.6 ):** May require manual verification  

---

## Performance Analysis

### Confusion Matrix (Test Set)

| Actual \ Predicted | Fake | Real |
| ------------------ | ---- | ---- |
| Fake               | 6733 | 226  |
| Real               | 176  | 5490 |

### Cross-Validation Results

* Accuracy: 96.53% ± 0.19%  
* AUC-ROC: 99.45% ± 0.05%  
* F1-Score: 96.16% ± 0.21%  

---

## License

This project is licensed under the MIT License.
