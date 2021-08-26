import pandas as pd
import numpy as np

from .baseRecommender.abstract_recommender import AbstractRecommender

class PopularityRecommender(AbstractRecommender):
    
    def __init__(self, name):
        super().__init__(name)
        
    def fit(self, ratings, movies = None, y = None):
        self.most_popular = ratings.groupby('movieId')['rating'].count().sort_values(ascending=False)
        self.ratings = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False)
        self.watched_by_user = ratings.set_index('userId')
        self.movies = movies[['title']]
        
        return self
        
    def predict(self, user_id, top_n=10, verbose=False, predict_rating = False):
        # Recommend the more popular items that the user hasn't seen yet.
        
        watched = self.watched_by_user.loc[user_id]['movieId']
        if top_n == -1:
            top_n = len(self.most_popular.index)
        recommendation= self.most_popular.drop(watched).iloc[:top_n].index.to_list()

        if verbose:
            if self.movies_names is None:
                raise Exception('Parameter `movie` is required in `verbose` mode')

            recommendation = self.movies[recommendation]
            
        if predict_rating:
            if verbose:
                recommendation['rating'] = self.ratings[recommendation]
            else:
                recommendation = self.ratings[recommendation]
                
        return recommendation

    def get_most_popular(self):
        return self.most_popular

        return recommendation