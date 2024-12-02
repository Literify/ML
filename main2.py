from flask import Flask, request, jsonify
import os
from PIL import Image
import io
from fitur1_ML import predict as predict_books
from fitur2_ml import predict_genre_book, recommend_books
import pytesseract
content_df="./content_df.csv"
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

@app.route("/predict-genre", methods=["POST"])
def predict_genre():
    try:
        # Ensure a file is uploaded
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400
        
        # Open the file as an image
        image = Image.open(io.BytesIO(file.read()))
        
        # Process the image and extract text
        text = pytesseract.image_to_string(image)
        
        # Predict genre using fitur2_ml
        genre = predict_genre_book(text)
        
        return jsonify({"extracted_text": text, "predicted_genre": genre})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend-books", methods=["POST"])
def recommend():
    try:
        user_id = request.json.get("user_id")
        genre = request.json.get("genre")
        if not user_id or not genre:
            return jsonify({"error": "Missing user_id or genre"}), 400

        # Load data and call the recommendation function
        filtered_books = content_df[content_df['genre'] == genre]
        recommendations = recommend_books(user_id, filtered_books)
        
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-books", methods=["POST"])
def predict_books_endpoint():
    try:
        # Ensure a file is uploaded
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Open the file as an image
        image = Image.open(io.BytesIO(file.read()))

        # Predict book-related features using fitur1_ML
        predictions = predict_books(image)
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8080, debug=True)
