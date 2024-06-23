import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

# Load the data
data = pd.read_csv('songdata.csv')
data = data[0:1000]
# Extract relevant columns
lyrics = data['text']

# Vectorize the lyrics using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(lyrics)

# Calculate the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the song that matches the title
    idx = data[data['song'] == title].index[0]

    # Get the pairwise similarity scores of all songs with that song
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the songs based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar songs
    sim_scores = sim_scores[1:11]

    # Get the song indices
    song_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar songs
    return data['song'].iloc[song_indices]

# Streamlit GUI
st.title('The Song Recommendation System')
st.title('Where words fail, music speaks.')
# User input for song selection
song_title = st.selectbox('Select a song:', data['song'])

# Display recommendations
if st.button('Show Recommendations'):
    recommendations = get_recommendations(song_title)
    st.write('Recommended songs:')
    for i, song in enumerate(recommendations):
        st.write(f"{i+1}. {song}")
