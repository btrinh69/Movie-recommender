import pandas as pd
import numpy as np

import numpy as np
import scipy
import pandas as pd
import math
import random

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class ModelEvaluator:
    
    def __init__(self, 
                 train_set,
                 test_set,
                 movies
                ):
        if 'userId' in train_set.columns:
            train_set.set_index('userId', inplace = True)
        if 'userId' in test_set.columns:
            test_set.set_index('userId', inplace = True)
        if 'movieId' in movies.columns:
            movies.set_index('userId', inplace = True)
        self.train_set = train_set
        self.test_set = test_set
        self.interacted_movies = train_set['movieId']
        self.test_movies = train_set['movieId']
        self.all_movies = set(movies.index)
        self.movie_id_to_name = movies['title'].to_dict()
        

    def _sample_not_watched_movies(self, user_id, sample_size = 100, random_state=0):
        interacted_items = set(self.interacted_movies.loc[user_id])
        non_interacted_items = self.all_movies.difference(interacted_items)
        
        random.seed(random_state)
        
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        
        return set(non_interacted_items_sample)
    

    def _hit_at_n(self, relevant_movies, prediction, n):
        return len(set(relevant_movies).intersection(set(prediction[:n])))

    def _recall_at_n(self, relevant_movies, num_hits, n):
        num_rel_movies = min(len(relevant_movies), n)
        
        return num_hits/num_rel_movies
    
    def _precision_at_n(self, num_hits, n):
        return num_hits/n
    
    def _auc_at_k(self, user_id, prediction, relevant_movies, k):
        not_watched_movies = self._sample_not_watched_movies(user_id)
        negative_examples = list(not_watched_movies - set(relevant_movies))
        random.shuffle(negative_examples)
        random.shuffle(relevant_movies)
        
        tp = 0
        fp = 0
        for pe in relevant_movies[:k]:
            for ne in negative_examples:
                try:
                    if prediction.loc[pe] > prediction.loc[ne]:
                        tp += 1
                except KeyError:
                    tp += 1
      
        return tp/(len(negative_examples) * min(k, len(relevant_movies)))

    def evaluate_model_for_user(self, model, user_id, top_n = 10, verbose = False):
        # Getting the items in test set
        relevant_movies = self.test_set.loc[user_id]
        # Only get movies with rating > 2
        relevant_movies = relevant_movies[relevant_movies['rating'] > 2]['movieId']
        relevant_movies = list(relevant_movies.unique())
        
        if len(relevant_movies) == 0:
            return {'hits_at_5_count':0, 
                    'hits_at_10_count':0, 
                    'interacted_count': 0,
                    'recall_at_5': 0,
                    'recall_at_10': 0,
                    'precision_at_5':0,
                    'precision_at_10':0,
                    'auc_at_10': 0
                   }

        # Getting a ranked recommendation list from a model for a given user
        prediction_score = model.predict(user_id, top_n = top_n, predict_rating = True)

        prediction = list(prediction_score.index)
        hits_at_5 = self._hit_at_n(relevant_movies, prediction, 5)
        hits_at_10 = self._hit_at_n(relevant_movies, prediction, 10)        

        indv_metrics = {'hits_at_5_count':hits_at_5, 
                        'hits_at_10_count':hits_at_10, 
                        'interacted_count': len(relevant_movies),
                        'recall_at_5': self._recall_at_n(relevant_movies, hits_at_5, 5),
                        'recall_at_10': self._recall_at_n(relevant_movies, hits_at_10, 10),
                        'precision_at_5': self._precision_at_n(hits_at_5, 5),
                        'precision_at_10': self._precision_at_n(hits_at_5, 10),
                        'auc_at_10': self._auc_at_k(user_id, prediction_score, relevant_movies, 10)
                       }
        if verbose:
            indent = '\t'
            print('Watched movies:')
            for movie in list(self.interacted_movies.loc[user_id]):
                print(indent + self.movie_id_to_name[movie])
            print('Relevant movies:')
            for movie in relevant_movies:
                print(indent + self.movie_id_to_name[movie])
                
            print()
            print('Recommendation:')
            for movie in prediction:
                print(indent + self.movie_id_to_name[movie])
        
        return indv_metrics
    

    def evaluate_model(self, model, top_n = -1, indv_metrics_df = False):
        #print('Running evaluation for users')
        all_user_metrics = []
        for idx, user_id in enumerate(list(self.test_set.index.unique().values)):
            indv_metrics = self.evaluate_model_for_user(model, user_id, top_n = top_n)  
            indv_metrics['_user_id'] = user_id
            all_user_metrics.append(indv_metrics)
        print('%d users processed' % idx)

        all_user_metrics_df = pd.DataFrame(all_user_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        agg_recall_at_5 = all_user_metrics_df['hits_at_5_count'].sum() / float(all_user_metrics_df['interacted_count'].sum())
        agg_recall_at_10 = all_user_metrics_df['hits_at_10_count'].sum() / float(all_user_metrics_df['interacted_count'].sum())
        agg_precision_at_5 = all_user_metrics_df['precision_at_5'].mean()
        agg_precision_at_10 = all_user_metrics_df['precision_at_10'].mean()
        agg_auc_at_10 = all_user_metrics_df['auc_at_10'].mean()
        
        agg_metrics = {'modelName': model.get_model_name(),
                       'recall_at_5': agg_recall_at_5,
                       'recall_at_10': agg_recall_at_10,
                       'agg_precision_at_5': agg_precision_at_5,
                       'agg_precision_at_10': agg_precision_at_10,
                       'agg_auc_at_10': agg_auc_at_10
                      }
        if indv_metrics_df == True:
            return agg_metrics, all_user_metrics_df
        return agg_metrics