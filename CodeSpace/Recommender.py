import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tensorrec

# Load the dataset
movies = pd.read_csv("C:/Users/phili/Downloads/Mini Project/Dataset/ml-25m/movies.csv")
ratings = pd.read_csv("C:/Users/phili/Downloads/Mini Project/Dataset/ml-25m/ratings.csv")

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(ratings, test_size=0.2)

# Content-based filtering
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Collaborative filtering
user_item_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating')
user_item_matrix = user_item_matrix.fillna(0)

# Create the TensorRec model
model = tensorrec.TensorRec(
    n_components=5,
    user_repr_graph=tensorrec.representation_graphs.DenseRepresentationGraph(),
    item_repr_graph=tensorrec.representation_graphs.DenseRepresentationGraph(),
    loss_graph=tensorrec.loss_graphs.WMRBLossGraph(),
    prediction_graph=tensorrec.prediction_graphs.CosineSimilarityPredictionGraph(),
    n_sampled_items=int(user_item_matrix.shape[1] * 0.01),
    user_side_info_graph=tensorrec.side_information_generators.RetrievalSideInfoGenerator(user_item_matrix)
)

# Train the model
user_ids = np.array(train_data['userId'].unique())
item_ids = np.array(train_data['movieId'].unique())
model.fit(user_ids, item_ids, user_item_matrix, epochs=10)

# Make recommendations for a user
def recommend_movies(user_id):
    user_index = np.where(user_ids == user_id)[0][0]
    user_ratings = user_item_matrix[user_index]
    known_positives = movies[movies['movieId'].isin(user_ratings[user_ratings > 0].index)]['title']
    scores = model.predict(user_ids[user_index], item_ids)
    top_items = movies.loc[np.argsort(-scores)]
    recommended_movies = top_items[~top_items['title'].isin(known_positives)]['title'][:10]
    return recommended_movies

# Test the model
for user_id in test_data['userId'].unique()[:10]:
    print('User ID:', user_id)
    print('Known Positives:', movies[movies['movieId'].isin(train_data[train_data['userId'] == user_id]['movieId'])]['title'].tolist())
    print('Recommended Movies:', recommend_movies(user_id).tolist())
    print('\n')
