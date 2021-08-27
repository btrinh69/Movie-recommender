import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

from .baseRecommender.abstract_recommender import AbstractRecommender

"""
Possible improvement:
- Item-based CF on the most recent movies
- Filter out movies rated low by the users
- Add more features to the movies similarity (E.g.: movies' description, director, main actor/actress, tags, etc.)
"""
class ItemBasedCollaborativeFiltering(AbstractRecommender):
    def __init__(self, name):
        super().__init__(name)
    
    def fit(self, ratings, movies, y = None):
        if not 'userId' in ratings.columns:
            ratings.reset_index(inplace = True)
        self.interaction = ratings.pivot_table(values = 'rating', 
                                  columns = 'movieId', 
                                  fill_value = 0, 
                                  index = 'userId')
        # Transpose to get movies correlation
        self.interaction = self.interaction.T
        # Calculate their cosine similarity
        self.similarities = cosine_similarity(self.interaction)
        
        self.similarities = pd.DataFrame(self.similarities,
                                         index = self.interaction.index,
                                         columns = self.interaction.index
                                        )
        
        self.movie_id_to_name = movies['title']
        self.user_watched_movies = ratings.set_index('userId')['movieId']
        self.movies_ratings = ratings[['movieId', 'rating']].groupby('movieId').mean()
        
        
        
    def predict(self, user_id, top_n=10, verbose=False, predict_rating = False, single_item:int = None):
        if single_item is None:
            watched_movies = list(self.user_watched_movies.loc[user_id])
        else:
            watched_movies = single_item
        
        # Use watched movies to generate top recommendations
        top_sim_movies = self.similarities.loc[watched_movies]
        
        # Drop watched movies similarity
        top_sim_movies = top_sim_movies.drop(watched_movies, axis = 1)

        # Get the top recommendation by getting the movies with the highest aggregate similarity to the movies this users has watched weighted by their mean rating by all users
        top_rec = (((top_sim_movies * np.array(self.movies_ratings.loc[watched_movies])).sum()) \
                   / (np.array(top_sim_movies.sum()+1))) \
        .sort_values(ascending = False).iloc[:top_n]
        
        # If verbse is True, output the name of the movies instead of their IDs
        if verbose:
            top_rec.index = self.movie_id_to_name.loc[top_rec.index].to_list()
        
        # If predict_rating is True, return the predicted rating by the user as well
        if predict_rating:
            return top_rec
        
        return top_rec.index.to_list()