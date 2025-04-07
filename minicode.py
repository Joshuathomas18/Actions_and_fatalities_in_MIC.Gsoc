import re
import spacy
import pandas as pd
import nltk
import gensim
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from collections import Counter
from summa.summarizer import summarize
from nltk.util import ngrams

# ✅ Load SpaCy model with dependency parsing
nlp = spacy.load("en_core_web_sm")

# ✅ Load Pre-trained Word2Vec (Use the correct path)
word_vectors = KeyedVectors.load_word2vec_format("C:/Users/User/Desktop/NLTK_vscode/GoogleNews-vectors-negative300.bin", binary=True)

# ✅ Download necessary NLTK resources
nltk.download("opinion_lexicon")
nltk.download("punkt")
from nltk.corpus import opinion_lexicon

# ✅ Sentiment word lists
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# ✅ Death-related keywords
death_keywords = {"killed", "deaths", "fatalities", "casualties", "dead", "attacks", "strikes"}
minor_injury_keywords = {"wounded", "injured", "hurt"}  # For max fatalities

# ✅ MIC-related bigrams/trigrams
mic_phrases = {
    "bigrams": {("military", "attack"), ("air", "strike"), ("border", "conflict")},
    "trigrams": {("cross", "border", "attack"), ("deadly", "air", "strike")}
}

# ✅ Load valid country list from CSV
country_df = pd.read_csv("C:/Users/User/Desktop/NLTK_vscode/states2016.csv")
valid_countries = set(country_df["statenme"].str.lower())

# ✅ Conflict-related words to filter relevant countries
conflict_keywords = {"attack", "war", "bombing", "strike", "troops", "invade"}

def extract_articles(text):
    """Split text into separate articles using '----' as a delimiter."""
    return text.split("----")

def preprocess_text(article):
    """Summarize, tokenize, and clean the text."""
    summary = summarize(article, ratio=0.3)  # Summarize to 30% of original text
    tokens = word_tokenize(summary.lower())  # Tokenize & convert to lowercase
    return tokens

def detect_deaths(tokens):
    """Check if any death-related words appear in the text."""
    return any(word in death_keywords for word in tokens)

def extract_countries(article):
    """Extract only the **2 most relevant** countries using NER, dependency parsing, and Word2Vec."""
    doc = nlp(article)
    country_mentions = Counter()

    # ✅ Step 1: Use Named Entity Recognition (NER)
    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text.lower() in valid_countries:
            country_mentions[ent.text.lower()] += 1  # Count occurrences

    # ✅ Step 2: Use Dependency Parsing to find countries linked to conflict words
    for token in doc:
        if token.dep_ in {"pobj", "dobj"} and token.head.text.lower() in conflict_keywords:
            if token.text.lower() in valid_countries:
                country_mentions[token.text.lower()] += 2  # Higher weight for context relevance

    # ✅ Step 3: Use Word2Vec for similarity with attack-related words
    for country in valid_countries:
        country_words = country.split()
        for word in country_words:
            if word in word_vectors:
                try:
                    similarity_scores = [
                        word_vectors.similarity(word, tok.text.lower())
                        for tok in doc
                        if tok.text.lower() in word_vectors
                    ]
                    if max(similarity_scores, default=0) > 0.7:
                        country_mentions[country] += 3  # Highest weight for Word2Vec relevance
                except KeyError:
                    continue  

    # ✅ Step 4: Keep only the **top 2 most mentioned countries**
    top_countries = [c for c, _ in country_mentions.most_common(2)]
    
    return ", ".join(top_countries) if top_countries else "Unknown"

def extract_date(article):
    """Extract event date using SpaCy's DATE entity."""
    doc = nlp(article)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    return dates[0] if dates else "Unknown"

def extract_sentiment(tokens):
    """Count positive, negative, and neutral words in the text."""
    pos_count = sum(1 for word in tokens if word in positive_words)
    neg_count = sum(1 for word in tokens if word in negative_words)
    neutral_count = len(tokens) - (pos_count + neg_count)
    return pos_count, neg_count, neutral_count

def detect_mic_ngrams(tokens):
    """Detect MIC-related bigrams and trigrams in the text."""
    bigram_set = set(ngrams(tokens, 2))
    trigram_set = set(ngrams(tokens, 3))
    
    has_mic_bigram = any(bigram in mic_phrases["bigrams"] for bigram in bigram_set)
    has_mic_trigram = any(trigram in mic_phrases["trigrams"] for trigram in trigram_set)

    return has_mic_bigram or has_mic_trigram

def classify_mic(min_fatalities, neg_words, mic_ngrams_detected):
    """
    Classify MIC status using fatalities, sentiment, and n-grams.
    
    - If min fatalities > 2 → MIC
    - If negative words > 5 and fatalities exist → MIC
    - If MIC-related n-grams are found → MIC
    - Otherwise → Not MIC
    """
    if min_fatalities > 2:
        return "MIC"
    if neg_words > 5 and min_fatalities > 0:
        return "MIC"
    if mic_ngrams_detected:
        return "MIC"
    return "Not MIC"

def process_document(text):
    """Process the document and extract key details."""
    articles = extract_articles(text)
    data = []

    for article in articles:
        tokens = preprocess_text(article)
        if detect_deaths(tokens):  # Process only if deaths are mentioned
            date = extract_date(article)
            countries = extract_countries(article)

            if countries == "Unknown":  # ✅ Remove rows where country is unknown
                continue  

            fatalities = Counter(tokens)
            min_fatalities = sum(fatalities[word] for word in death_keywords)
            max_fatalities = min_fatalities + sum(fatalities[word] for word in minor_injury_keywords)

            # Sentiment features
            pos_count, neg_count, neu_count = extract_sentiment(tokens)

            # N-Gram detection
            mic_ngrams_detected = detect_mic_ngrams(tokens)

            # MIC Classification
            mic_status = classify_mic(min_fatalities, neg_count, mic_ngrams_detected)

            data.append([date, min_fatalities, max_fatalities, countries, pos_count, neg_count, neu_count, mic_status])

    return pd.DataFrame(data, columns=["Date", "Min Fatalities", "Max Fatalities", "Countries Involved", "Positive Words", "Negative Words", "Neutral Words", "MIC Status"])

# ✅ Read input text file and process
with open("C:/Users/User/Desktop/NLTK_vscode/New York Times/2002-2010/2002.txt", "r", encoding="ISO-8859-1") as f:
    text = f.read()

df = process_document(text)

# ✅ Save results
df.to_csv("deathmic_output.csv", index=False)
print("✅ Processing complete. Output saved to deathmic_output.csv")
