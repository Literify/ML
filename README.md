# Book Recommendation and Genre Prediction Models

This repository contains two machine learning models for predicting book genres and providing personalized book recommendations. The models are built using Python and TensorFlow and are designed to work with preprocessed book data and user preferences.

---

## Features

### 1. **Genre Prediction** (`fitur2_ml.py`)
- Uses a Bidirectional LSTM model for text classification to predict book genres.
- Supports OCR-based text extraction from book covers or other images using Tesseract.
- Incorporates preprocessing techniques, including stopword removal and text vectorization.
- Outputs the predicted genre based on the provided text input.

### 2. **Recommendation System** (`fitur1_ML.py` and `fitur2_ml.py`)
- Provides personalized book recommendations based on user interactions and preferences.
- Features two approaches:
  - Collaborative filtering using TensorFlow Recommenders (`fitur2_ml.py`).
  - Content-based filtering using cosine similarity (`fitur1_ML.py`).
- Supports custom weights for recommendation tuning.
- Outputs a list of recommended books with detailed metadata.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Literify/ML/book-recommendation-genre-prediction.git
   cd book-recommendation-genre-prediction
2. Install dependencies:
   Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Ensure the following files are in the working directory:
   - sampled_categories.csv
   - content_df.csv
   - user_ids.csv
   - label_encoder.pkl
   - vectorizer_vocab.pkl
   - cosine_similarity.pkl
   - Pretrained model weights (model_genre_classification_weights.h5, model_recomendation_weights.h5).
  
## Usage
