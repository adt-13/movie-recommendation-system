# app.py

import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# SET PAGE CONFIG AS THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Movie Recommender", layout="wide")

# --- Part 1: Data Loading and Model Preparation ---
# This part of the code will only run once, thanks to Streamlit's cache.
@st.cache_resource
def load_data_and_train_models():
    """
    This function loads all necessary data and trains the recommendation models.
    """
    # Load the datasets from the 'ml-latest-small' folder
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    
    # --- Collaborative Filtering (CF) Model ---
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    cf_model = SVD(n_factors=100, n_epochs=20, random_state=42)
    cf_model.fit(trainset)
    
    # --- Content-Based (CB) Model ---
    movies_df['genres_processed'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_processed'])
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Create a mapping from movieId to matrix index
    movieid_to_idx = pd.Series(movies_df.index, index=movies_df['movieId'])
    
    return ratings_df, movies_df, cf_model, cosine_sim_matrix, movieid_to_idx

# Load everything
ratings_df, movies_df, cf_model, cosine_sim_matrix, movieid_to_idx = load_data_and_train_models()

# --- Part 2: Hybrid Recommendation Function ---
def get_hybrid_recommendations(user_id, top_n=10):
    """
    Generates top N recommendations for a user using a hybrid approach.
    """
    # Get all movie IDs
    all_movie_ids = movies_df['movieId'].unique()
    
    # Get movies the user has already rated
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
    
    # Get movies to predict (unrated ones)
    movies_to_predict_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    # Find the user's top-rated movie to act as a 'profile'
    try:
        user_top_movie_id = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4.0)].sort_values('rating', ascending=False).iloc[0]['movieId']
        top_movie_idx = movieid_to_idx[user_top_movie_id]
    except IndexError:
        # Handle case where user has no high ratings
        return []

    recommendations = []
    
    for movie_id in movies_to_predict_ids:
        if movie_id in movieid_to_idx:
            # Get CF score
            cf_score = cf_model.predict(user_id, movie_id).est

            # Get CB score
            movie_idx = movieid_to_idx[movie_id]
            cb_score = cosine_sim_matrix[top_movie_idx][movie_idx]
            
            # Normalize CB score to a 0-5 scale
            cb_score_normalized = cb_score * 5.0
            
            # Calculate hybrid score with a 70/30 weight
            hybrid_score = (cf_score * 0.7) + (cb_score_normalized * 0.3)
            recommendations.append((movie_id, hybrid_score))

    # Sort recommendations
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N movie titles and scores
    top_recommendations = []
    for movie_id, score in recommendations[:top_n]:
        title = movies_df.loc[movies_df['movieId'] == movie_id, 'title'].iloc[0]
        top_recommendations.append((title, score))
        
    return top_recommendations

# --- Part 3: Streamlit User Interface (UI) ---

st.title("🎬 Hybrid Movie Recommendation System")
st.write("This system uses different types of information to recommend movies: your past ratings (Collaborative) and movie genres (Content-Based).")

# Get a list of user IDs for the dropdown
user_ids = ratings_df['userId'].unique().tolist()
selected_user_id = st.selectbox("Select a User ID to get recommendations for:", user_ids)

if st.button("Get Movie Recommendations"):
    with st.spinner('Calculating recommendations...'):
        recommendations = get_hybrid_recommendations(user_id=selected_user_id, top_n=10)
        
        if recommendations:
            st.success(f"Top 10 Recommendations for User {selected_user_id}:")
            
            # Display recommendations
            for i, (title, score) in enumerate(recommendations):
                st.write(f"**{i+1}. {title}** (Predicted Score: {score:.2f})")
        else:
            st.error("Could not generate recommendations. This user may not have enough highly-rated movies to create a profile. Please try another user.")