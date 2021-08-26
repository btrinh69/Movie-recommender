import pandas as pd
import numpy as np

from .baseRecommender.abstract_recommender import AbstractRecommender

from lightfm import LightFM
from lightfm.data import Dataset

class LightFMRecommender(AbstractRecommender):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name)

        self.model = LightFM(*args, **kwargs)

        
    def _split_year(self, title):
        x = title.strip()
        if len(x) > 6 and x[-6] == "(" and x[-1] == ")" and x[-5:-1].isdigit():
            return x[:-6].strip(), int(x[-5:-1])
        return x, np.nan
    
    def _split_year_df(self, movies):
        movies['year'] = [np.nan]*len(movies.index)
        movies[['title', 'year']] = movies['title'].apply(self._split_year).to_list()
        movies['year'] = movies['year'].fillna(int(movies['year'].mode()))
        movies['year'] = movies['year'].astype(int)
        
        return movies
        
    # Might separate this function to a standalone transformer
    # so that we can break
    def _build_dataset(self, ratings, movies, include_item_features = True):
        movies = self._split_year_df(movies)
        
        movies['embedding_features'] = movies['year'].apply(lambda x: [x]) + movies['genres'].str.split('|')
        
        all_genres = []
        for genre_list in list(movies['genres'].str.split('|')):
            all_genres += genre_list

        all_genres = list(set(all_genres))
        all_year = list(set(movies['year']))
        
        dataset = Dataset()
        dataset.fit(ratings['userId'],
                    movies['movieId'],
                    item_features = all_genres + all_year
                   )
        
        (interactions, weights) = dataset.build_interactions((
            (x[1]['userId'], x[1]['movieId'])
            for x in ratings[['userId', 'movieId', 'rating']].iterrows())
        )
        if include_item_features:
            item_features = dataset.build_item_features(((x[1]['movieId'], x[1]['embedding_features'])
                                                     for x in movies.iterrows()))
        else:
            item_features = None
            
        self.dataset = dataset
        self.interactions = interactions
        self.weights = weights
        self.item_features = item_features
        
        
    # TODO: add base year
    def fit(self, ratings, movies, include_item_features = True, *args, **kwargs):
        ratings_copy = ratings.copy()
        if 'userId' not in ratings.columns:
            ratings_copy.reset_index(inplace = True)
        
        movies_copy = movies.copy()
        if 'movieId' not in movies.columns:
            movies_copy.reset_index(inplace = True)
        
        
        self.movies_id_to_name = movies_copy[['movieId', 'title']].set_index('movieId')
        
        self._build_dataset(ratings_copy, movies_copy, include_item_features = include_item_features)
        
        self.model.fit(self.interactions, 
                       item_features = self.item_features)
        
        return self
    
    def predict(self, user_id, top_n = 10, verbose = False, predict_rating = False):
        internal_user_id = self.dataset.mapping()[0][user_id]
        scores = self.model.predict(internal_user_id, 
            np.array(list((self.dataset.mapping()[2].values()))),
            item_features = self.item_features
        )
        
        prediction = pd.DataFrame({
            'movieId': list(self.dataset.mapping()[2].keys()),
            'score': scores
        })
        
        prediction = prediction.sort_values(by = 'score', ascending = False)
        
        if verbose:
            prediction['movieId'] = prediction['movieId'].apply(lambda x: self.movies_id_to_name.loc[x])
            
        if predict_rating:
            return prediction.set_index('movieId')['score'].iloc[:top_n]
        
        return list(prediction['movieId'].iloc[:top_n])
        