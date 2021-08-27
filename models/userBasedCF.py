import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

from .baseRecommender.abstract_recommender import AbstractRecommender

"""
Possible improvement:
- Add more features to the users similarity (E.g.: users' biography, movie preference, search activities, tags interactions, etc.)
"""
class UserBasedCollaborativeFiltering(AbstractRecommender):
    def __init__(self, name, k):
        super().__init__(name)
        self.k = k
    
    def fit(self, ratings, movies, y = None):
        if not 'userId' in ratings.columns:
            ratings.reset_index(inplace = True)
            
        # Pivot to get the users' interactions with all movies
        self.interaction = ratings.pivot_table(values = 'rating', 
                                  columns = 'movieId', 
                                  fill_value = 0, 
                                  index = 'userId')
        # Calculate cosine similarities between users
        self.similarities = cosine_similarity(self.interaction)
        
        self.similarities = pd.DataFrame(self.similarities,
                                         index = self.interaction.index,
                                         columns = self.interaction.index
                                        )
        
        self.movie_id_to_name = movies['title']
        
        self.ratings = ratings.set_index('userId')['movieId']
        
    def predict(self, user_id, top_n=10, verbose=False, predict_rating = False):
        watched_movies = list(self.ratings.loc[user_id])
        # Get the top `k` similar users
        top_sim_user = self.similarities.loc[user_id].sort_values(ascending = False).iloc[:self.k]
        top_sim_interaction = self.interaction.loc[top_sim_user.index.to_list()]
        
        # Get the top recommendation by getting the movies with the highest rating weighted by users' similarity
        top_rec = (((top_sim_interaction.T * top_sim_user).T).sum()/(top_sim_user.sum()+1)).sort_values(ascending = False)
        top_rec = top_rec.drop(watched_movies).iloc[:top_n]
        
        # If verbse is True, output the name of the movies instead of their IDs
        if verbose:
            top_rec.index = self.movie_id_to_name.loc[top_rec.index].to_list()
        
        # If predict_rating is True, return the predicted rating by the user as well
        if predict_rating:
            return top_rec
        
        return top_rec.index.to_list()