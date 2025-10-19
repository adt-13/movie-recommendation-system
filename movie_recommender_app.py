import pandas as pd
import streamlit as st
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_data_and_prepare_model():
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    links_df = pd.read_csv('ml-latest-small/links.csv')

    movie_rating_counts = ratings_df.groupby('movieId').size().reset_index(name='rating_count')
    
    popularity_threshold = 50
    popular_movie_ids = movie_rating_counts[movie_rating_counts['rating_count'] >= popularity_threshold]['movieId']
    
    movies_df = movies_df[movies_df['movieId'].isin(popular_movie_ids)]
    
    movies_df = pd.merge(movies_df, links_df, on='movieId', how='left')
    
    movies_df.reset_index(drop=True, inplace=True)
    
    movies_df['genres_processed'] = movies_df['genres'].str.replace('|', ' ', regex=False).fillna('')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_processed']).astype(np.float32)
    
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    return movies_df, cosine_sim_matrix

movies_df, cosine_sim_matrix = load_data_and_prepare_model()

def fetch_poster(tmdb_id):
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

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to your favorites based on their genres.")

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
