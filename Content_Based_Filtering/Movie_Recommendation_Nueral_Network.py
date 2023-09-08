import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Step 1: Data Collection & Preprocessing

# Read the dataset
df = pd.read_csv('movies.csv')
print(df.head())

# For simplicity, let's consider only the genres
df['genres'] = df['genres'].apply(lambda x: x.split('|'))

# Encoding genres
mlb = MultiLabelBinarizer()
encoded_genres = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(encoded_genres, columns=mlb.classes_)

# Assuming you have some user ratings data for movies. For this example, I'll randomly generate some user ratings
np.random.seed(42)
user_ratings = np.random.randint(1, 6, size=(df.shape[0], 1))

# Step 3: Neural Network Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(genre_df.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')  # predicting rating, so linear activation
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Training
X_train, X_test, y_train, y_test = train_test_split(genre_df.values, user_ratings, test_size=0.2, random_state=42)


# Reshape the labels to match the expected shape
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(type(X_train), type(y_train))


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 5: Evaluation (just a basic one in this case)
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Step 6: Recommendation
# For demonstration, let's say user_id is 5 and we want to recommend top 5 movies based on predicted ratings
user_id = 5
predicted_ratings = model.predict(genre_df.values)
top_5_movies = np.argsort(predicted_ratings[:, 0])[-5:]
recommended_movies = df.iloc[top_5_movies]

print("Recommended Movies:")
print(recommended_movies[['title', 'genres']])
