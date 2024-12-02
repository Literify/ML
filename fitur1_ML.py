import os
import pickle
#from typing import Dict, Text
#import ipywidgets as widgets
#import random
#from IPython.display import display, HTML
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
#import tensorflow as tf
#from tensorflow.keras.preprocessing.text import tokenizer_from_json
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
#from tensorflow.keras.optimizers import Adam
#import regex as re
#import string
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
#os.environ['TF_USE_LEGACY_KERAS'] = '1'
#from PIL import Image
#import pytesseract

sampled_categories = pd.read_csv('sampled_categories.csv')
content_df = pd.read_csv('content_df.csv')
user_ids_df = pd.read_csv('user_ids.csv')


def predict(title, data, cos_sim, similarity_weight=0.7, top_n=10):
    index_movie = data[data['book_title'] == title].index
    similarity = cos_sim[index_movie].T

    sim_df = pd.DataFrame(similarity, columns=['similarity'])
    final_df = pd.concat([data, sim_df], axis=1)

    final_df['final_score'] = final_df['weighted_average']*(1-similarity_weight) + final_df['similarity']*similarity_weight

    final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
    final_df_sorted_show = final_df_sorted[['book_title', 'description', 'authors', 'genre', 'publisher', 'Price', 'image', 'previewLink', 'infoLink']]
    return print(final_df_sorted_show.to_json(orient='records', lines=False)))

# Load the cosine similarity matrix from pickle
cos_sim = pickle.load(open('cosine_similarity.pkl', 'rb'))
# Convert the cosine similarity matrix to a dense format
cos_sim_dense = cos_sim.toarray()
predict("It's Not All Song and Dance: A Life Behind the Scenes in the Performing Arts", content_df, cos_sim_dense, similarity_weight=0.7, top_n=3)



