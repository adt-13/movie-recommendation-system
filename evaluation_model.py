import pandas as pd
import streamlit as st
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Movie Recommender", layout="wide")

@st.cache_resource
def load_data_and_train_models():
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    cf_model = SVD(n_factors=100, n_epochs=20, random_state=42)
    cf_model.fit(trainset)
    
    movies_df['genres_processed'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres_processed'])
    cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    movieid_to_idx = pd.Series(movies_df.index, index=movies_df['movieId'])
    
    return ratings_df, movies_df, cf_model, cosine_sim_matrix, movieid_to_idx

ratings_df, movies_df, cf_model, cosine_sim_matrix, movieid_to_idx = load_data_and_train_models()

def get_hybrid_recommendations(user_id, top_n=10):
    all_movie_ids = movies_df['movieId'].unique()
    
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].values

    movies_to_predict_ids = [movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids]
    
    try:
        user_top_movie_id = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['rating'] >= 4.0)].sort_values('rating', ascending=False).iloc[0]['movieId']
        top_movie_idx = movieid_to_idx[user_top_movie_id]
    except IndexError:
        return []

    recommendations = []
    
    for movie_id in movies_to_predict_ids:
        if movie_id in movieid_to_idx:
            cf_score = cf_model.predict(user_id, movie_id).est

            movie_idx = movieid_to_idx[movie_id]
            cb_score = cosine_sim_matrix[top_movie_idx][movie_idx]
            
            cb_score_normalized = cb_score * 5.0
            
            hybrid_score = (cf_score * 0.7) + (cb_score_normalized * 0.3)
            recommendations.append((movie_id, hybrid_score))
            
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    top_recommendations = []
    for movie_id, score in recommendations[:top_n]:
        title = movies_df.loc[movies_df['movieId'] == movie_id, 'title'].iloc[0]
        top_recommendations.append((title, score))
        
    return top_recommendations

st.title("🎬 Hybrid Movie Recommendation System")
st.write("This system uses different types of information to recommend movies: your past ratings (Collaborative) and movie genres (Content-Based).")

user_ids = ratings_df['userId'].unique().tolist()
selected_user_id = st.selectbox("Select a User ID to get recommendations for:", user_ids)

if st.button("Get Movie Recommendations"):
    with st.spinner('Calculating recommendations...'):
        recommendations = get_hybrid_recommendations(user_id=selected_user_id, top_n=10)
        
        if recommendations:
            st.success(f"Top 10 Recommendations for User {selected_user_id}:")
            
            for i, (title, score) in enumerate(recommendations):
                st.write(f"**{i+1}. {title}** (Predicted Score: {score:.2f})")
        else:
            st.error("Could not generate recommendations. This user may not have enough highly-rated movies to create a profile. Please try another user.")
