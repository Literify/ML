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

### 1. Genre Prediction
Run the script `fitur2_ml.py` for genre prediction:
```bash
python fitur2_ml.py --input "path/to/input_text.txt"
```
Replace `path/to/input_text.txt` with the path to your text input file. The script will output the predicted genre.

### 2. Recommendation System
For book recommendations, you can use either of the two scripts:

#### Collaborative Filtering
```bash
python fitur2_ml.py --recommend "path/to/user_data.json"
```
Replace `path/to/user_data.json` with the path to your user data file. The script will generate personalized recommendations.

#### Content-Based Filtering
```bash
python fitur1_ml.py --content "path/to/book_metadata.json"
```
Replace `path/to/book_metadata.json` with the path to the book metadata file. The script will recommend books based on content similarity.

### 3. Testing the Application
For those who want to try the application directly, a pre-built `.apk` file can be provided. Contact the development team for access to the APK and instructions on how to install it on your Android device.


---

## File Structure
```
.
├── fitur1_ml.py                # Script for content-based recommendation
├── fitur2_ml.py                # Script for genre prediction and collaborative filtering
├── requirements.txt            # Dependencies
├── model_genre_classification_weights.h5  # Pretrained weights for genre prediction
├── model_recomendation_weights.h5        # Pretrained weights for recommendation system
├── sampled_categories.csv      # Sampled book categories
├── content_df.csv              # Content data for books
├── user_ids.csv                # User ID data
├── label_encoder.pkl           # Label encoder for genre prediction
├── vectorizer_vocab.pkl        # Text vectorizer vocabulary
├── cosine_similarity.pkl       # Precomputed cosine similarity matrix
└── README.md                   # Documentation
```

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear and concise messages.
4. Submit a pull request for review.

---

## Acknowledgments
- **TensorFlow Recommenders** for the collaborative filtering implementation.
- **Tesseract OCR** for text extraction.
- Kaggle datasets for providing valuable book data for training and evaluation.

Feel free to reach out if you encounter any issues or have feature suggestions!

