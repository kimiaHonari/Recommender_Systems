import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the data
newsgroups = fetch_20newsgroups(subset='all')
texts = newsgroups.data
labels = newsgroups.target

# 2. Preprocess the data
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=500)

x_train, x_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# 3. Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=500),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(20, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Extract embeddings
embedding_layer = model.layers[0]
weights = embedding_layer.get_weights()[0]


def get_article_vector(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, padding='post', maxlen=500)
    return model.predict(padded)


def recommend_articles(input_text, n=5):
    # Assuming get_article_vector returns 2D array (e.g., shape (1, embedding_size))
    input_vec = get_article_vector(input_text).reshape(1, -1)

    # If get_article_vector returns 2D arrays, concatenate them along the first axis
    all_article_vecs = np.concatenate([get_article_vector(text) for text in texts], axis=0)

    # Compute cosine similarity
    cosine_sims = cosine_similarity(input_vec, all_article_vecs)

    # Get top n articles indices
    recommended_idxs = cosine_sims[0].argsort()[-n:][::-1]

    return [texts[i] for i in recommended_idxs]


# Test recommendation
recommended = recommend_articles("Machine learning and artificial intelligence in finance")
print(recommended)
