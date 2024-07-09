import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

# Load data
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on="title")

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Remove rows with missing values
movies.dropna(inplace=True)

# Helper function to convert stringified list of dictionaries to list of strings
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
            count += 1
        else:
            break   
    return L  

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

def convert_overview(obj):
    result = obj.split()
    return result

def remove_space(obj):
    L = []
    for i in obj:
        i = i.replace(" ", "")
        L.append(i)
    return L 

# Apply conversions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(convert_overview)
movies['keywords'] = movies['keywords'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)

# Create tags column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# New dataframe with selected columns
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list to string
def listToString(obj):
    str1 = " "
    return (str1.join(obj))

new_df['tags'] = new_df['tags'].apply(listToString)

# Convert tags to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Initialize CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# Initialize PorterStemmer
ps = PorterStemmer()

def stem(text):
    str1 = ""
    for i in text.split():
        str1 += ps.stem(i) + " "
    return str1.strip()

# Apply stemming
new_df['tags'] = new_df['tags'].apply(stem)

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distance = similarity[index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:10]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

# Example usage
recommend("The Avengers")
pickle.dump(new_df,open('movie.pkl','wb'))
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('movie_similarity.pkl','wb'))