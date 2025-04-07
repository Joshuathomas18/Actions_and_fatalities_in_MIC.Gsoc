<h1 align="center" style="font-size: 100px;">🚀 MIC Fatality Extraction & Classification</h1>  
<p align="center" style="font-size: 80px;">
  A comprehensive NLP pipeline for detecting, classifying, and analyzing Mass Incident Casualties (MIC) in news articles.
</p>  

## 📌 Overview  

Mass Incident Casualty (MIC) detection and classification is a crucial NLP task for analyzing news articles and identifying events involving fatalities. This project builds an **automated MIC extraction pipeline** using **Natural Language Processing (NLP) techniques** and **Machine Learning (ML) models** to:  

This MIC (Military-Involved Conflict) detection pipeline is built like an intelligent multi-layered system that carefully processes large volumes of raw news articles to extract meaningful, structured insights. It starts by **recursively scanning `.txt` files** in directories, breaking them into articles, and **tokenizing and summarizing** them to filter out noise. Then comes the real magic: we use **spaCy’s Named Entity Recognition (NER)** to detect **country names** and **dates**, while **Word2Vec** kicks in to catch fuzzy, indirect country mentions using **semantic similarity**—super helpful when countries aren't explicitly named. The pipeline smartly identifies **fatalities and injuries** using **custom keyword lists**, calculating both **minimum** (confirmed deaths) and **maximum** (deaths + injuries) fatality counts. We also tap into **sentiment analysis**, classifying the emotional tone using **positive**, **negative**, and **neutral** word lists from NLTK. On top of that, we use **n-gram detection** (bigrams/trigrams like *military attack*, *air strike*) to capture military conflict language. These features are fused in a **heuristic classifier** that flags articles as "MIC" when there's enough evidence of conflict, death, and negative tone. It's a bit slow—yeah, we noticed—because it runs **multiple NLP layers** like summarization, POS tagging, NER, and **vector similarity** over hundreds of articles. But honestly? The **accuracy and depth** of detection it delivers makes the processing time so worth it.

---

Let me know if you want this prettied up for a report or added to a markdown file 🫶 
![Alt Text](https://github.com/Joshuathomas18/Actions_and_fatalities_in_MIC.Gsoc/blob/main/Screenshot%202025-04-05%20130936.png)
**This is an Example of the output of the code**
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
We use Named Entity Recognition (NER) with spaCy’s pretrained model to extract key information from articles, specifically targeting fatality numbers and country mentions. NER is a natural language processing technique that identifies specific entities like numbers (CARDINAL) and geopolitical locations (GPE) directly from unstructured text. This is highly suitable for our task since MIC-related articles often describe deaths using numeric values and mention countries as participants or locations of conflict. By combining NER with keyword filtering (e.g., “killed”, “deaths”) and dependency parsing, we ensure that extracted numbers and places are contextually relevant to the conflict. Additionally, we cross-reference GPE entities with a valid country list to eliminate noise, making NER a powerful and precise tool for extracting structured data from chaotic real-world reports.
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


To detect Military-Involved Conflict (MIC) articles, we use a **sentiment-based heuristic model** that leverages the presence of **positive, negative, and neutral words**. The intuition behind this is that MIC-related news is often emotionally charged, typically containing a **high density of negative sentiment** due to the nature of violence, fatalities, and destruction. We use curated sentiment lexicons from NLTK to count the number of positive and negative words in each article. Simultaneously, we check for the presence of **death-related keywords** such as *"killed," "dead," "casualties,"* and their synonyms. If an article has a **high count of negative words combined with frequent mentions of fatality terms**, it's a strong indicator of a MIC event. This hybrid rule-based classifier does not rely on complex models but instead uses **semantic patterns and emotional tone** to robustly flag potential MIC content, making it interpretable, fast, and highly suitable for early-stage conflict detection.

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

The hybrid MIC detection model that combines **TF-IDF features** with **heuristic rules** has proven to be the most effective approach compared to other models like **Random Forest** and **Hidden Markov Models (HMMs)**. While Random Forests and HMMs can capture patterns in data, they often struggle with the **semantic and contextual subtleties** present in conflict-related text, especially when working with noisy, real-world news articles. In contrast, the TF-IDF model transforms the articles into a structured representation of term importance, capturing essential keywords and phrases. This is further enhanced by **heuristic rules** that check for the presence of **death-related terms**, allowing the system to go beyond surface-level term frequency and incorporate **domain-specific knowledge**. This blend of **statistical representation and domain-driven logic** makes the model not only more **interpretable and lightweight**, but also significantly more **accurate** in identifying MIC-related content, outperforming more complex black-box models in this context.
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


##  **🖨 Sample Output & Performance Notes** 
The following is an example of the structured output generated by our pipeline. It captures critical information such as dates, locations, fatality counts, sentiment scores, and ultimately classifies each article as either MIC (Mass Incident Casualty) or Not MIC.

This output is the result of a layered, rule-driven NLP system rather than a fine-tuned pretrained model. Because we're not leveraging existing large-scale classifiers and instead building a domain-specific heuristic model from scratch, the pipeline involves multiple processing stages — including tokenization, summarization, Named Entity Recognition (NER), sentiment analysis, Word2Vec-based similarity checks, and n-gram detection.

These operations, while highly interpretable and tailored for the MIC detection task, are computationally expensive and input-intensive — especially when applied recursively across hundreds of raw text files. As a result, the overall processing time is higher compared to models that rely purely on vectorized or pretrained embeddings. However, this trade-off ensures greater transparency, control, and adaptability, making our approach ideal for early-stage or exploratory conflict detection tasks where explainability is key.
[2002.txt](deathmic_output.csv) shows how the model performs for a textfile in this case "2002 news articles".
Here is the [**CODE**](minicode.py) for the same.


This model operates on a heuristic-based NLP pipeline, which isn't just calling a pre-trained model or fine-tuning something lightweight. It's built from scratch with layers of custom logic for MIC detection, entity recognition, fatality classification, and sentiment analysis.
Because of this—and the fact that NLP processing can be computationally expensive, especially for non-optimized, rule-heavy models—the processing time is quite high.
Also, the dataset itself is massive: 766 text files, each with potentially unstructured and complex language. Running deep NLP analysis on this scale takes time.
So here's a  the [**Output**](final_MIC_output.csv) I was able to generate within the current processing window 👇


![Alt Text](https://github.com/Joshuathomas18/Actions_and_fatalities_in_MIC.Gsoc/blob/main/Screenshot%202025-04-07%20124246.png)







