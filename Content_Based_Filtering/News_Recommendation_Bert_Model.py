import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, TFBertModel
import tensorflow

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_article_vector(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = bert_model(inputs)
    # Use the CLS token representation as the article embedding
    return outputs['last_hidden_state'][:, 0, :].numpy()

# The rest of your code remains largely the same...

# 1. Load the data
newsgroups = fetch_20newsgroups(subset='all')
texts = newsgroups.data
labels = newsgroups.target

# Generate embeddings for all articles
all_article_vecs = np.concatenate([get_article_vector(text) for text in texts], axis=0)

def recommend_articles(input_text, n=5):
    input_vec = get_article_vector(input_text)
    cosine_sims = cosine_similarity(input_vec.reshape(1, -1), all_article_vecs)
    recommended_idxs = cosine_sims[0].argsort()[-n:][::-1]
    return [texts[i] for i in recommended_idxs]

def format_recommendation(recommended_articles):
    formatted_results = []

    for article in recommended_articles:
        # Extracting the 'Subject' field from the header
        subject_line = next((line for line in article.split('\n') if line.startswith('Subject: ')), None)
        subject = subject_line.split('Subject: ')[1] if subject_line else "No Subject"

        # Removing headers and keeping only the main body
        body_start_idx = article.find('\n\n') + 2
        body = article[body_start_idx:].strip()

        # Truncate body for better display, if needed
        max_body_length = 100
        if len(body) > max_body_length:
            body = body[:max_body_length] + "..."

        formatted_results.append(f"Subject: {subject}\n{body}\n{'-'*50}")

    return "\n".join(formatted_results)

recommended = recommend_articles("Machine learning and artificial intelligence in finance")
print(format_recommendation(recommended))

