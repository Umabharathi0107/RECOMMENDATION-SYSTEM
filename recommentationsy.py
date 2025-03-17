import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

# Load sample dataset (MovieLens 100K dataset)
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv(url, sep='\t', names=column_names, engine='python')

# Define a reader for Surprise library
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Use Singular Value Decomposition (SVD) for matrix factorization
model = SVD()

# Perform cross-validation
cross_validate(model, dataset, cv=5, verbose=True)

# Train model on full dataset
trainset = dataset.build_full_trainset()
model.fit(trainset)

# Function to recommend top N items for a user
def recommend_items(user_id, model, trainset, n=5):
    all_items = set(trainset.all_items())
    rated_items = {j for (j, _) in trainset.ur[user_id]}
    unrated_items = all_items - rated_items
    
    predictions = [(item, model.predict(user_id, item).est) for item in unrated_items]
    recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    return recommendations

# Example: Recommend 5 movies for user 1
user_id = 1
recommended_movies = recommend_items(user_id, model, trainset)
print(f"Top 5 recommendations for User {user_id}: {recommended_movies}")
