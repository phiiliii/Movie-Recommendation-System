#PREPROCESSING OF DATASET
#______________________________________________
import pandas as pd

movies = pd.read_csv("C:/Users/phili/Downloads/Mini Project/Dataset/ml-25m/movies.csv")
ratings = pd.read_csv("C:/Users/phili/Downloads/Mini Project/Dataset/ml-25m/ratings.csv")

movie_ratings = pd.merge(movies, ratings, on='movieId')

#COLLABORATIVE FILTERING
#_______________________________________________

from surprise import Dataset
from surprise import Reader
from surprise import KNNBasic
from surprise.model_selection import train_test_split

# Load data into surprise format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(movie_ratings[['userId', 'movieId', 'rating']], reader)

# Split data into training and testing sets
trainset, testset = train_test_split(data, test_size=.25)

# Build a user-based collaborative filtering model
sim_options = {'name': 'cosine',
               'user_based': True}
algo = KNNBasic(sim_options=sim_options)
algo.fit(trainset)
