<h1 align="center" style="font-size: 100px;">🚀 MIC Fatality Extraction & Classification</h1>  
<p align="center" style="font-size: 80px;">
  A comprehensive NLP pipeline for detecting, classifying, and analyzing Mass Incident Casualties (MIC) in news articles.
</p>  

## 📌 Overview  

Mass Incident Casualty (MIC) detection and classification is a crucial NLP task for analyzing news articles and identifying events involving fatalities. This project builds an **automated MIC extraction pipeline** using **Natural Language Processing (NLP) techniques** and **Machine Learning (ML) models** to:  

- Extract fatalities (minimum and maximum) from news text.  
- Identify the countries involved in each incident.  
- Analyze sentiment (positive, negative, neutral words) for better classification.  
- Accurately classify articles as **MIC** or **Not MIC** using advanced **heuristic methods**, **POS-based extraction**, and **n-grams**.  

### 🛠 Tech Stack & Libraries  

This project utilizes:  

- **Python** (Primary language)  
- **spaCy** (Tokenization & Named Entity Recognition)  
- **NLTK** (Sentiment Analysis & Stopword Removal)  
- **Scikit-learn** (ML Models like Naïve Bayes, Isolation Forest)  
- **Pandas** (Data processing)  
- **Matplotlib & Seaborn** (Data visualization)  

 # 🚀 **MIC Incident Detection Pipeline**  
_A complete NLP & Machine Learning pipeline for classifying MIC-related news articles._

---

## 🔄 **Pipeline Overview**  

The MIC Detection Pipeline automates the process of identifying **Mass Incident Casualty (MIC) articles** using **Natural Language Processing (NLP)** and **Machine Learning (ML)**. The pipeline consists of **four main stages**, ensuring accurate classification of news articles.

---

## 🏗 **Pipeline Stages**  

### 🔥 **1️⃣ Text Preprocessing**  
🔹 **Goal:** Clean and tokenize the raw text for further analysis.  
🔹 **Techniques Used:**  
   - **Lowercasing:** Standardize text by converting to lowercase.  
   - **Punctuation Removal:** Eliminate special characters.  
   - **Stopwords Removal:** Remove common words (e.g., "the", "is").  
   - **Tokenization:** Break text into individual words.  
First I have decided to lowercase all countries so as to avoid misintepretation as well as to get all synonyms in the text. Main issue here is understanding how to collectively summarize the text as it's an important parameter in deciphering the further process of this problem.Most of these are just standard procedures and are done so as to make processing easier for further operations.
🔹 **Code Snippet:**  
```python
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    tokens = text.split()  # Tokenize
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(tokens)

# Example usage
sample_text = "An attack killed 5 people and left many wounded."
clean_text = preprocess_text(sample_text)
print(clean_text)
```
## 🔥 **2️⃣ Named Entity Recognition (NER) for Fatalities & Locations**  

### 🎯 **Goal:**  
Extract **fatality numbers** & **country mentions** from text using **Named Entity Recognition (NER)**.

### 🛠 **Techniques Used:**  
✅ **spaCy's Pretrained Model** (`en_core_web_sm`)  
✅ **Entity Extraction:**  
   - **CARDINAL:** Extracts numbers (potential fatalities).  
   - **GPE (Geopolitical Entity):** Extracts country names.  

---

### 📝 **How it Works?**  
1️⃣ The **NER model** scans the article text.  
2️⃣ It **identifies** and **extracts** numbers & country mentions.  
3️⃣ Fatalities & locations are stored as structured data.  

---

### 💻 **Code Snippet:**  
```python
import spacy

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    """
    Extracts fatality numbers and country mentions from text.
    """
    doc = nlp(text)
    fatalities = []
    countries = []

    for ent in doc.ents:
        if ent.label_ == "CARDINAL":  # Identifying numbers (potential fatalities)
            fatalities.append(ent.text)
        elif ent.label_ == "GPE":  # Identifying country mentions
            countries.append(ent.text)

    return fatalities, list(set(countries))  # Removing duplicate countries

# Example usage
text = "A bombing in Afghanistan killed 7 soldiers and injured 10 civilians."
fatalities, countries = extract_entities(text)

print(f"Fatalities: {fatalities}")
print(f"Countries: {countries}")
```

## 🔥 **3️⃣ Sentiment & Death Word Analysis for MIC Classification**  

### 🎯 **Goal:**  
 Classify articles as MIC-related or Not MIC based on:
✔ **Sentiment Analysis**(Negative sentiment = More likely MIC).
✔ **Death-Word Thresholding** (Frequent mentions of death-related words)..

### 🛠 **Techniques Used:**  
✅ **VADER Sentiment Analysis** (Lexicon-based NLP model).
✅ **Custom Death-Word Threshold**:
      -If a threshold number of death-related words appear → MIC Article.
      -Otherwise → Not MIC.


### 📝 **How it Works?**  
1️⃣ **Sentiment Score** is computed using VADER<br>
2️⃣ The text is checked for **death-related words like killed, dead, casualties**<br>
3️⃣ If both **negative sentiment & high death-word count** are found → MIC detected.  



```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Death-related words
death_keywords = {"killed", "dead", "fatalities", "deaths", "massacre", "bombing"}

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

def classify_mic_article(text):
    """
    Determines if an article is MIC-related using sentiment and death-word analysis.
    """
    # Compute sentiment score
    sentiment_score = analyzer.polarity_scores(text)["compound"]

    # Count death-related words
    death_word_count = sum(1 for word in text.split() if word.lower() in death_keywords)

    # MIC Classification Criteria
    if sentiment_score < -0.5 and death_word_count >= 2:
        return "MIC"
    else:
        return "Not MIC"

# Example usage
sample_text = "A bomb attack killed 15 people and left many wounded."
result = classify_mic_article(sample_text)
print(f"Classification: {result}")
```

##  **🎯 4️⃣ Classification & MIC Detection**  

### 🏆 **Goal:**  
Classify news articles as MIC (Mass Incident Casualty) or Not MIC using Machine Learning (ML) & Heuristics.

### 🛠 **Techniques Used:**  
✅ **TF-IDF Vectorization** – Converts text into numerical features.<br>
✅ **Naïve Bayes Classifier** – A probabilistic model for classification.<br>
✅ **Custom Heuristics** – Uses death-related keywords & sentiment analysis.


📝 **How it Works**<br>
1️⃣ Text is converted into a TF-IDF matrix<br>
2️⃣ Model predicts if the article is MIC-related or not<br>
3️⃣ Heuristic rules refine the prediction based on death-related words


```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Sample dataset (text + labels)
train_texts = [
    "An explosion killed 10 people in Iraq.",
    "A sports event was held in Germany.",
    "A terrorist attack injured 15 civilians in India.",
    "A new tech conference is happening in the USA."
]
train_labels = [1, 0, 1, 0]  # 1 = MIC, 0 = Not MIC

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)

# Train Naïve Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, train_labels)

# Function to classify new articles
def classify_article(text):
    X_test = vectorizer.transform([text])
    prediction = classifier.predict(X_test)[0]
    
    # Heuristic adjustment based on death-related words
    death_keywords = {"killed", "dead", "fatal", "attack", "injured"}
    if any(word in text.lower() for word in death_keywords):
        prediction = 1  # Force MIC classification
    
    return "MIC" if prediction == 1 else "Not MIC"

# Example usage
sample_text = "A massive earthquake killed 50 people."
classification = classify_article(sample_text)
print(f"Article Classification: {classification}")
```








