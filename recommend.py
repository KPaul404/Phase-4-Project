import streamlit as st
import pandas as pd
import pickle
import requests
from surprise import SVD, Dataset, Reader

# Function to fetch movie poster from TMDB
def fetch_poster(movie_id):
    if movie_id is None:
        print("No movie ID provided")
        return "https://via.placeholder.com/500x750?text=Poster+Not+Available"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=326761804d34fcdb563b7641d0194cd7&language=en-US"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data from TMDB: Status Code {response.status_code}")
        return "https://via.placeholder.com/500x750?text=Poster+Not+Available"
    data = response.json()
    poster_path = data.get('poster_path', '')
    if poster_path:
        poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return poster_url
    else:
        print("Poster path not found in TMDB response")
        return "https://via.placeholder.com/500x750?text=Poster+Not+Available"

# Function to recommend movies using SVD model
def recommend_svd(user_id, ratings_df, movies_with_links_df, model, num_recommendations=5):
    all_movies = ratings_df['movieId'].unique()
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    unrated_movies = [movie for movie in all_movies if movie not in rated_movies]
    predictions = [(movie, model.predict(user_id, movie).est) for movie in unrated_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_recommendations = predictions[:num_recommendations]

    recommended_movie_titles = []
    recommended_movie_ids = []
    for movie_id, _ in top_recommendations:
        movie_row = movies_with_links_df[movies_with_links_df['movieId'] == movie_id]
        recommended_movie_titles.append(movie_row['title'].iloc[0])
        recommended_movie_ids.append(movie_row['tmdbId'].iloc[0])

    return recommended_movie_titles, recommended_movie_ids

# Load data
model = pickle.load(open("trained_svd_model.pkl", "rb"))
movies_df = pd.read_csv(r"ml-latest-small/movies.csv")
ratings_df = pd.read_csv(r"ml-latest-small/ratings.csv")
links_df = pd.read_csv(r"ml-latest-small/links.csv")

# Merge TMDB IDs into the movies dataframe
movies_with_links_df = movies_df.merge(links_df, on='movieId', how='left')

# Streamlit UI
st.title("Group 4 Movie Recommender")
user_id_input = st.number_input("Enter your user ID", min_value=1, step=1)
num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Recommend"):
    movie_titles, movie_ids = recommend_svd(user_id_input, ratings_df, movies_with_links_df, model, num_recommendations)
    cols = st.columns(num_recommendations)
    for idx, (title, tmdb_id) in enumerate(zip(movie_titles, movie_ids)):
        poster_url = fetch_poster(tmdb_id)
        with cols[idx]:
            st.text(title)
            st.image(poster_url)