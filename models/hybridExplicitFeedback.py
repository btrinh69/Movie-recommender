import pandas as pd
import numpy as np

from .baseRecommender.abstract_recommender import AbstractRecommender

from inspect import signature
import logging

class HybridRecommender(AbstractRecommender):
    def __init__(self, name, recommenders, weights, logger = None):
        super().__init__(name)
        
        if len(recommenders) != len(weights):
            raise Exception('The len of `recommenders` and `weights` mismatch')
            
        if len(recommenders) < 2:
            raise Exception('There is only 1 recommenders. Need 2 or more to combine')
            
        for recomender in recommenders:
            if not 'predict_rating' in signature(recomender.predict).parameters.keys():
                raise Exception(f'Recommender {recommender.get_model_name()} does not support rating prediction. Cannot combine recommenders that have no rating prediction')
            
        self.recommenders = recommenders
        self.weights = weights
        
        if logger is None:
            self.logger = logging
        
    def fit(self,
            pretrained = False,
            movies = None,
            ratings = None
           ):
        if not pretrained and not isinstance(movies, pd.DataFrame) and not isinstance(ratings, pd.DataFrame):
            raise Exception('`pretrained` is False and `movies` and `ratings` are both None. Models must either be pretrained or provided with training data (`movies` and `ratings`)')
            
        self.movies_id_to_name = movies['title']
        self.num_movies = len(movies.index)
        self.total_weights = sum(self.weights)
            
        if not pretrained:
            for recommender in self.recommenders:
                self.logger.info(f'Fitting model: {recommender.get_model_name()}')
                recommender.fit(ratings, movies)
                
        self.logger.info('Done training')
                
    def predict(self, user_id, top_n = 10, verbose = False, predict_rating = False):
        top_n_by_models = dict()
        for i in range(len(self.recommenders)):
            top_n_by_models[self.recommenders[i].get_model_name()] = self.recommenders[i].predict(user_id, top_n = self.num_movies, predict_rating = True)
            top_n_by_models[self.recommenders[i].get_model_name()] *= self.weights[i]
            
        model_list = list(top_n_by_models.keys())
        top_n_by_models_df = pd.DataFrame(top_n_by_models[model_list[0]])
        
        for model in model_list:
            top_n_by_models_df = top_n_by_models_df.join(pd.DataFrame(top_n_by_models[model]), rsuffix = f'{model}_')
            
        prediction_agg = top_n_by_models_df.iloc[:, 0]
        for col in range(1, len(top_n_by_models_df.columns)):
            prediction_agg += top_n_by_models_df.iloc[:, col]
            
        prediction_agg /= self.total_weights
        
        prediction_agg = prediction_agg.sort_values(ascending = False).iloc[:top_n]
        
        if verbose:
            prediction_agg.index = self.movies_id_to_name[prediction_agg.index]
        
        if predict_rating:
            return prediction_agg
        
        return prediction_agg.index.to_list()