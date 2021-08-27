import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

from .baseRecommender.abstract_recommender import AbstractRecommender

# Global fixed number to partially standardized
YEAR_DIFF_SEMI_STANDARDIZE = 10

"""
Possible improvement:
- Add more features to the movies similarity (E.g.: movies' description, director, main actor/actress, tags, average_rating, etc.)
"""
class ContentBasedRecommender(AbstractRecommender):
    
    def __init__(self, name, genre_list, year_diff_semi_standardize = YEAR_DIFF_SEMI_STANDARDIZE):
        super().__init__(name)
        self.genre_list = genre_list
        self.year_diff_semi_standardize = year_diff_semi_standardize
        
    # Helper function to calculate movies' similarity by genres
    def _genre_similarity(self, X, movie_id_name_mapping):
        genre_similarity = cosine_similarity(X)
        genre_similarity = pd.DataFrame(genre_similarity, 
                                        columns = movie_id_name_mapping.index.to_list(), 
                                        index = movie_id_name_mapping.index.to_list())
        return genre_similarity
    
    # Helper function to calculate movies' similarity by year
    def _year_similarity(self, X):
        X['dummy_dim'] = 1
        year_similarity_flat = pdist(X.to_numpy(), 'euclidean')
        year_similarity = squareform(year_similarity_flat)
        year_similarity = np.exp(-year_similarity/self.year_diff_semi_standardize)
        
        return year_similarity
    
    # Helper function to split the movies' year from their name
    def _split_year(self, title):
        x = title.strip()
        if len(x) > 6 and x[-6] == "(" and x[-1] == ")" and x[-5:-1].isdigit():
            return title, int(x[-5:-1])
        return title, np.nan

    # Helper function to preprocess the data into suitable format to calculate the similarity
    def _preprocess(self, movies, genre_list):
        # Generate the year column
        movies['year'] = [np.nan]*len(movies.index)

        movies[['title', 'year']] = movies['title'].apply(self._split_year).to_list()

        movies['year'] = movies['year'].fillna(int(movies['year'].mode()))

        movies['year'] = movies['year'].astype(int)
        
        # Generate genres one-hot features for each movie
        for genre in genre_list:
            movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

        movies.drop('genres', axis = 1, inplace = True)

        if 'movieId' in movies.columns:
            movies.set_index('movieId', inplace = True)

        return movies
        
    def fit(self, ratings, movies, y = None, copy = True):
        if copy:
            movies_copy = movies.copy()
            self.ratings = ratings.copy()
        else:
            movies_copy = movies
            self.ratings = ratings
            
        movies_copy = self._preprocess(movies_copy, self.genre_list)
        if 'userId' in ratings.columns:
            self.ratings.set_index('userId', inplace = True)
        movie_id_name_mapping = movies[['title']]
        movies_copy.drop('title', axis = 1, inplace = True)

        # Generate similarity by genres
        genre_similarity = self._genre_similarity(movies_copy.drop('year', axis = 1), movie_id_name_mapping)
        # Generate similarity by year
        year_similarity = self._year_similarity(movies_copy[['year']].copy())
        
        # Combine the similarity
        self.similarities = year_similarity * genre_similarity
        self.movie_id_to_name = movies['title']
                
        return self
        
    def predict(self, user_id, top_n=10, verbose=False, predict_rating = False):
        # Onely recommend the movies that the user has not watched
        watched_movies = self.ratings.loc[user_id]['movieId']

        # Get the ratings of movies to use as weights for the similarity
        rates = self.ratings.loc[user_id][['movieId', 'rating']].set_index('movieId')

        # Get the similarities between the users' rated movies and all the remaining
        top_sim = self.similarities.loc[watched_movies].copy()
        # Calculate the similarity weighted by ratings
        top_sim = ((top_sim.T * rates['rating']).T.sum()/(np.array(top_sim.sum())+1))
        
        # Remove watched films
        top_sim.drop(rates.index, inplace = True)
        top_sim = top_sim.sort_values(ascending = False).iloc[:top_n]
        
        # If verbse is True, output the name of the movies instead of their IDs
        if verbose:
            top_sim.index = self.movie_id_to_name.loc[top_sim.index].to_list()
        
        # If predict_rating is True, return the predicted rating by the user as well
        if predict_rating:
            return top_sim
        
        return top_sim.index.to_list()
        


