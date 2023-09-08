import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from scipy.sparse import coo_matrix
from tensorflow.contrib.factorization import WALSModel


# Load the MovieLens dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Define user_id
user_id = 1

# Re-index userId and movieId
ratings['userId'] = ratings['userId'].astype("category")
ratings['movieId'] = ratings['movieId'].astype("category")

# Create a user-movie matrix
data = coo_matrix((ratings['rating'].astype(np.float32),
                  (ratings['userId'].cat.codes.copy(), ratings['movieId'].cat.codes.copy())))

num_users = data.shape[0]
num_items = data.shape[1]

# Create WALS model
n_components = 10  # or whatever number of components you'd like to use
model = WALSModel(num_users, num_items, n_components=n_components, regularization=0.01, row_weights=None, col_weights=None)

# Define input tensors
row_factor = tf.convert_to_tensor(model.row_factors)
col_factor = tf.convert_to_tensor(model.col_factors)

# Define and train WALS model
input_tensor = tf.SparseTensor(indices=np.mat([data.row, data.col]).transpose(),
                               values=data.data, dense_shape=data.shape)

row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
col_update_op = model.update_col_factors(sp_input=input_tensor)[1]

with tf.Session() as sess:
    sess.run(model.initialize_op)
    sess.run(model.worker_init)
    for _ in range(10):  # Number of iterations
        sess.run([row_update_op, col_update_op])

    user_factors = row_factor.eval()
    item_factors = col_factor.eval()

# Get recommendations for the user
user_representation = user_factors[ratings[ratings['userId'] == user_id]['userId'].cat.codes.iloc[0]]
scores = item_factors.dot(user_representation)

recommended_item_indices = scores.argsort()[-10:][::-1]
for item_idx in recommended_item_indices:
    movie_id = ratings['movieId'].cat.categories[item_idx]
    print(movies[movies['movieId'] == movie_id]['title'].iloc[0], "-", scores[item_idx])
