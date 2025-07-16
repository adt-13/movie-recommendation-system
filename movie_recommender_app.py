# app.py

import pandas as pd
import streamlit as st
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set Page Config
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Part 1: Data Loading and Model Preparation ---
@st.cache_resource
def load_data_and_prepare_model():
    """
    Loads data, filters for popular movies to save memory,
    and prepares the content-based model.
    """
    # Load data
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    links_df = pd.read_csv('ml-latest-small/links.csv')

    # --- NEW: Filter for popular movies ---
    # Calculate how many ratings each movie has
    movie_rating_counts = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
    
    # Define a threshold for what counts as a popular movie
    popularity_threshold = 50
    popular_movie_ids = movie_rating_counts[movie_rating_counts['rating_count'] >= popularity_threshold]['movieId']
    
    # Filter the movies_df and ratings_df to keep only popular movies
    movies_df = movies_df[movies_df['movieId'].isin(popular_movie_ids)]
    
    # --- End of new filtering section ---
    
    # Merge with links_df to get tmdbId
    movies_df = pd.merge(movies_df, links_df, on='movieId', how='left')
    
    # We need to reset the index after filtering so that matrix indices match
    movies_df.reset_index(drop=True, inplace=True)
    
    # Content-Based Model
    movies_df['genres_processed'] = movies_df['genres'].str.replace('|', ' ', regex=False).fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_processed']).astype(np.float32)
    
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return movies_df, cosine_sim_matrix

# Load data and prepare the model
movies_df, cosine_sim_matrix = load_data_and_prepare_model()


# --- Part 2: Functions to Fetch Posters and Recommendations ---
def fetch_poster(tmdb_id):
    """Fetches the movie poster URL from TMDB API."""
    if pd.isna(tmdb_id):
        return "https://via.placeholder.com/500x750.png?text=No+Poster"
        
    api_key = "7ab0640e5591d896d2d28ec32c63a80c"
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://via.placeholder.com/500x750.png?text=No+Poster"
    except requests.exceptions.RequestException:
        return "https://via.placeholder.com/500x750.png?text=API+Error"

def get_movie_recommendations(title, top_n=10):
    """Generates top N similar movies and fetches their posters."""
    try:
        # Find the index of the movie that matches the title in our filtered dataframe
        idx = movies_df[movies_df['title'] == title].index[0]
    except IndexError:
        return [], []

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    
    recommended_df = movies_df.iloc[movie_indices]
    recommended_titles = recommended_df['title'].tolist()
    recommended_posters = recommended_df['tmdbId'].apply(fetch_poster).tolist()
    
    return recommended_titles, recommended_posters

# --- Part 3: Streamlit User Interface (UI) ---
st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorites based on their genres.")

# The list of movies is now smaller and contains only popular ones
movie_titles = movies_df['title'].tolist()
selected_movie_title = st.selectbox(
    "Search for a movie to get recommendations:",
    movie_titles
)

if st.button("Get Recommendations"):
    if selected_movie_title:
        with st.spinner('Finding similar movies and fetching posters...'):
            rec_titles, rec_posters = get_movie_recommendations(title=selected_movie_title, top_n=5)
            
            if rec_titles:
                st.success(f"Movies similar to '{selected_movie_title}':")
                
                cols = st.columns(5)
                for i in range(len(rec_titles)):
                    with cols[i]:
                        st.image(rec_posters[i])
                        st.caption(rec_titles[i])