import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.contrib.factorization import WALSMatrixFactorization

# Ensure you're using TensorFlow 1.x
if not tf.__version__.startswith('1'):
    raise ValueError('This code requires TensorFlow V1.x')

# Load the dataset
data = pd.read_csv('lastfm_data.tsv', sep='\t', header=None, error_bad_lines=False)
data.columns = ['userId', 'timestamp', 'artistId', 'artistName', 'songId', 'songName']

# Create implicit feedback
data['playCount'] = 1
grouped_data = data.groupby(['userId', 'songId']).playCount.sum().unstack().fillna(0)
grouped_data_binary = grouped_data.applymap(lambda x: 1 if x > 0 else 0)

# Convert dataframe to sparse matrix format
user_encoder = LabelEncoder()
song_encoder = LabelEncoder()
row = user_encoder.fit_transform(grouped_data_binary.index)
col = song_encoder.fit_transform(grouped_data_binary.columns)
data_vals = grouped_data_binary.values.flatten()

sparse_matrix = coo_matrix((data_vals, (row, col)), shape=(len(set(row)), len(set(col))))
input_tensor = tf.SparseTensor(
    indices=np.array([sparse_matrix.row, sparse_matrix.col]).T,
    values=sparse_matrix.data.astype(np.float32),
    dense_shape=sparse_matrix.shape
)

num_users, num_songs = sparse_matrix.shape

# Hyperparameters
num_factors = 10  
regularization = 0.01

# Create WALS model
model = WALSMatrixFactorization(
    num_rows=num_users,
    num_cols=num_songs,
    embedding_dimension=num_factors,
    regularization=regularization,
    unobserved_weight=0
)

# Training
train_op = model.minimize(input_tensor)
num_iterations = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(num_iterations):
        sess.run(train_op)
    user_factors, song_factors = sess.run([model.row_factors, model.col_factors])

# Recommendations
def recommend_songs_for_user(user_id, num_recommendations=10):
    user_vector = user_factors[user_id]
    scores = song_factors.dot(user_vector)
    top_songs = scores.argsort()[-num_recommendations:][::-1]
    return top_songs
