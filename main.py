# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv(r"C:\Users\Derrick Baalaboore\Desktop\Movie-Recommendation-System-Dataset-main\movies.csv")

# Preprocess data
df.drop_duplicates(inplace=True)
df = df.pivot_table(index='title', columns='genres', values='rating')

# Split data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2)

# Calculate user similarity
user_similarity = cosine_similarity(train_data.fillna(0))

# Make movie recommendations for users
def get_movie_recommendations(user_id):
    user_ratings = df[user_id]
    similar_users = user_similarity[user_id-1]
    similar_users_indices = np.argsort(similar_users)[::-1][1:]
    recommended_movies = []
    for i in range(len(df.columns)):
        if np.isnan(user_ratings[i]):
            movie_rating = 0
            count = 0
            for j in similar_users_indices:
                if not np.isnan(train_data.iloc[i, j]):
                    movie_rating += train_data.iloc[i, j] * similar_users[j]
                    count += similar_users[j]
                if count > 0:
                    movie_rating /= count
            recommended_movies.append((df.columns[i], movie_rating))
    recommended_movies = sorted(recommended_movies, key=lambda x: x[1], reverse=True)[:10]
    return recommended_movies

# Test the model with new user ratings
user_ratings = [5, np.nan, 3, np.nan, 4, np.nan, np.nan, 2, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4]
df.loc['New User'] = user_ratings
recommended_movies = get_movie_recommendations(len(df.columns))
print(recommended_movies)
