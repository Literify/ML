# Import Library
import io
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from feature import fitur1_ML, fitur2_ml
import os
import random
import pytesseract
from PIL import Image
from fitur1_ML import predict as predict_books_1
from fitur2_ml import load_and_predict_genre, recommend_books, extract_text_from_image

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow_recommenders as tfrs

# Import Data
content_df = pd.read_csv("./data/content_df.csv")
sampled_categories = pd.read_csv('./data/sampled_categories.csv')
sampled_categories = sampled_categories.rename(columns={'review/score': 'user_rating', 'User_id': 'user_id', 'Title': 'book_title'})
user_ids_df = pd.read_csv('./data/user_ids.csv')

unique_books_df = sampled_categories[['book_title']].drop_duplicates()
ratings = tf.data.Dataset.from_tensor_slices(dict(sampled_categories[['user_id', 'book_title', 'user_rating']]))
books = tf.data.Dataset.from_tensor_slices(dict(unique_books_df[['book_title']]))
ratings = ratings.map(lambda x: {
    "book_title": x["book_title"],
    "user_id": x["user_id"],
    "rating": float(x["user_rating"])
})
books = books.map(lambda x: x["book_title"])
book_titles = books.batch(1_000)
user_ids = ratings.batch(1_000).map(lambda x: x["user_id"])
unique_book_titles = np.unique(np.concatenate(list(book_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

# Import Model
cos_sim = pickle.load(open('./model/cosine_similarity.pkl', 'rb'))
cos_sim_dense = cos_sim.toarray()
model_weights_path = './model/model_genre_classification_weights.h5'
vectorizer_vocab_path = '/content/model/vectorizer_vocab.pkl'
label_encoder_path = "./model/label_encoder.pkl"
model_weights_path = './model/model_recomendation_weights.h5'

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

@app.route("/recommend-book-fitur-1", methods=["POST"])
def recommend_book_fitur_1():
    try:
        # Get the title from the JSON request
        title = request.json.get("title")
        if not title:
            return jsonify({"error": "Title is required"}), 400
        
        # Call the predict function
        result = predict_books_1(title, data=content_df, cos_sim_dense, similarity_weight=0.7, top_n=3)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-genre", methods=["POST"])
def predict_genre():
    try:
        # Ensure a file is uploaded
        image_path = request.json.get("image_url")
        if not image_path:
            return jsonify({"error": "No image URL provided"}), 400
        
        # Download the image from the URL
        response = requests.get(image_path)
        if response.status_code != 200:
            return jsonify({"error": "Failed to retrieve the image from the URL"}), 400
        
        # Extract text from the image using Tesseract
        extracted_text = pytesseract.image_to_string(Image.open(io.BytesIO(response.content)))
        preprocessed_text = ' '.join(extracted_text.split())
        
        # Predict genre using fitur2_ml
        predicted_genre = load_and_predict_genre(preprocessed_text,
                                                 model_weights_path=model_weights_path,
                                                 vectorizer_vocab_path=vectorizer_vocab_path,
                                                 label_encoder_path=label_encoder_path) 
        
        return jsonify({"extracted_text": preprocessed_text, "predicted_genre": predicted_genre})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend-books-fitur-2", methods=["POST"])
def recommend_books_fitur_2():
    try:
        predicted_genre_input = request.json.get("predicted_genre")
        choice_genre_input = request.json.get("choice_genre")
        if not predicted_genre_input or not choice_genre_input:
            return jsonify({"error": "Missing predicted_genre or choice_genre"}), 400

        # Recommendation function
        recommendations = recommend_books(user_ids_df, content_df, predicted_genre_input, unique_book_titles, unique_user_ids, 
                                          model_weights_path=model_weights_path, top_n=3, choice=choice_genre_input)
        
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == "__main__":
    app.run(port=8080, debug=True)
