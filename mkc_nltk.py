import os
import re
import spacy
import pandas as pd
import nltk
import gensim
from tqdm import tqdm
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from collections import Counter
from summa.summarizer import summarize
from nltk.util import ngrams

# ✅ Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# ✅ Load Pre-trained Word2Vec (update your local path!)
word_vectors = KeyedVectors.load_word2vec_format(
    "C:/Users/User/Desktop/NLTK_vscode/GoogleNews-vectors-negative300.bin", binary=True
)

# ✅ Download required NLTK data
nltk.download("opinion_lexicon")
nltk.download("punkt")
from nltk.corpus import opinion_lexicon

# ✅ Word lists
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())
death_keywords = {"killed", "deaths", "fatalities", "casualties", "dead", "attacks", "strikes"}
minor_injury_keywords = {"wounded", "injured", "hurt"}
mic_phrases = {
    "bigrams": {("military", "attack"), ("air", "strike"), ("border", "conflict")},
    "trigrams": {("cross", "border", "attack"), ("deadly", "air", "strike")},
}
conflict_keywords = {"attack", "war", "bombing", "strike", "troops", "invade"}

# ✅ Load country CSV
country_df = pd.read_csv("C:/Users/User/Desktop/NLTK_vscode/states2016.csv")
valid_countries = set(country_df["statenme"].str.lower())

def extract_articles(text):
    return text.split("----")

def preprocess_text(article):
    summary = summarize(article, ratio=0.3)
    return word_tokenize(summary.lower())

def detect_deaths(tokens):
    return any(word in death_keywords for word in tokens)

def extract_countries(article):
    doc = nlp(article)
    country_mentions = Counter()

    for ent in doc.ents:
        if ent.label_ == "GPE" and ent.text.lower() in valid_countries:
            country_mentions[ent.text.lower()] += 1

    for token in doc:
        if token.dep_ in {"pobj", "dobj"} and token.head.text.lower() in conflict_keywords:
            if token.text.lower() in valid_countries:
                country_mentions[token.text.lower()] += 2

    for country in valid_countries:
        country_words = country.split()
        for word in country_words:
            if word in word_vectors:
                try:
                    similarity_scores = [
                        word_vectors.similarity(word, tok.text.lower())
                        for tok in doc if tok.text.lower() in word_vectors
                    ]
                    if max(similarity_scores, default=0) > 0.7:
                        country_mentions[country] += 3
                except KeyError:
                    continue

    top_countries = [c for c, _ in country_mentions.most_common(2)]
    return ", ".join(top_countries) if top_countries else "Unknown"

def extract_date(article):
    doc = nlp(article)
    dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
    return dates[0] if dates else "Unknown"

def extract_sentiment(tokens):
    pos = sum(1 for w in tokens if w in positive_words)
    neg = sum(1 for w in tokens if w in negative_words)
    neu = len(tokens) - (pos + neg)
    return pos, neg, neu

def detect_mic_ngrams(tokens):
    bigram_set = set(ngrams(tokens, 2))
    trigram_set = set(ngrams(tokens, 3))
    return any(b in mic_phrases["bigrams"] for b in bigram_set) or any(t in mic_phrases["trigrams"] for t in trigram_set)

def classify_mic(min_fatalities, neg_words, mic_ngrams_detected):
    if min_fatalities > 2:
        return "MIC"
    if neg_words > 5 and min_fatalities > 0:
        return "MIC"
    if mic_ngrams_detected:
        return "MIC"
    return "Not MIC"

def process_document(text):
    articles = extract_articles(text)
    rows = []
    for article in articles:
        tokens = preprocess_text(article)
        if detect_deaths(tokens):
            date = extract_date(article)
            countries = extract_countries(article)
            if countries == "Unknown":
                continue

            word_counts = Counter(tokens)
            min_fatalities = sum(word_counts[w] for w in death_keywords)
            max_fatalities = min_fatalities + sum(word_counts[w] for w in minor_injury_keywords)
            pos, neg, neu = extract_sentiment(tokens)
            mic_ngrams_detected = detect_mic_ngrams(tokens)
            mic_status = classify_mic(min_fatalities, neg, mic_ngrams_detected)

            rows.append([date, min_fatalities, max_fatalities, countries, pos, neg, neu, mic_status])
    return pd.DataFrame(rows, columns=[
        "Date", "Min Fatalities", "Max Fatalities", "Countries Involved",
        "Positive Words", "Negative Words", "Neutral Words", "MIC Status"
    ])

# ✅ Read all .txt files from directory with progress bar
def process_directory(directory_path, output_csv_path):
    file_list = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".txt"):
                file_list.append(os.path.join(root, file))

    total_files = len(file_list)
    with tqdm(total=total_files, desc="📂 Processing Files", ncols=100) as pbar:
        for full_path in file_list:
            file = os.path.basename(full_path)
            try:
                with open(full_path, "r", encoding="ISO-8859-1") as f:
                    text = f.read()
                    df = process_document(text)
                    if not df.empty:
                        df["Source File"] = file
                        if os.path.exists(output_csv_path):
                            df.to_csv(output_csv_path, mode='a', header=False, index=False)
                        else:
                            df.to_csv(output_csv_path, mode='w', header=True, index=False)
                pbar.set_postfix({"Last File": file})
            except Exception as e:
                print(f"❌ Error in {file}: {e}")
            pbar.update(1)

# ✅ Run full pipeline with progress and batching
if __name__ == "__main__":
    input_dir = "C:/Users/User/Desktop/NLTK_vscode/New York Times"
    output_csv = "final_MIC_output.csv"
    process_directory(input_dir, output_csv)
    print(f"\n✅ All done! Output saved to {output_csv}")
