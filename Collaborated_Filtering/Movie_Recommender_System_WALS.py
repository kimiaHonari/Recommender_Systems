import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares

# Load the MovieLens dataset (you need to download it and adjust the path accordingly)
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Create a user-movie matrix
data = coo_matrix((ratings['rating'].astype(np.float32),
                  (ratings['movieId'], ratings['userId'])))

# Initialize the ALS model
model = AlternatingLeastSquares(factors=50, regularization=0.01, iterations=50)

# Train the model
model.fit(data)

# Recommend movies for a user
user_id = 5  # Change this to any user ID from the dataset
recommended = model.recommend(user_id, data.tocsr(), N=10)

# Print the recommended movies
for movie_id, score in recommended:
    print(movies[movies['movieId'] == movie_id]['title'].iloc[0], "-", score)
