import pickle
import streamlit as st
import requests
import pandas as pd 

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/28?api_key=7293d59094afb4579b4b58226a26eeb4&language=en-US"
    data = requests.get(url,verify=False)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie, movies_data, similarity):
    index = movies_data.index[movies_data['title'] == movie][0]  # Find index of selected movie
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # Fetch the movie poster
        movie_id = movies_data.iloc[i[0]].movie_id  # Assuming movies_data is a DataFrame
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies_data.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

st.header('Movie Recommender System')

movie_dict=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movie_dict)

similarity=pickle.load(open('movie_similarity.pkl','rb'))

# Create dropdown menu with movie titles
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie, movies, similarity)
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
