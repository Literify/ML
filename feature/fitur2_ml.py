import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pytesseract
from PIL import Image

os.environ['TF_USE_LEGACY_KERAS'] = '1'

## Text Classification
def load_and_predict_genre(text, model_weights_path, vectorizer_vocab_path, label_encoder_path):
    # Parameters
    VOCAB_SIZE = 10000
    MAX_LENGTH = 32
    PADDING_TYPE = 'pre'
    TRUNC_TYPE = 'post'
    EMBEDDING_DIM = 16
    LSTM_DIM = 32
    DENSE_DIM = 24

    # Model definition
    model_lstm = tf.keras.Sequential([
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM)),
        tf.keras.layers.Dense(DENSE_DIM, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Load the model weights
    model_lstm.load_weights(model_weights_path)

    # Load the label encoder
    with open(label_encoder_path, 'rb') as file:
        label_encoder = pickle.load(file)

    # Load the saved vocabulary
    with open(vectorizer_vocab_path, 'rb') as file:
        vocabulary = pickle.load(file)

    # Recreate the TextVectorization layer using the loaded vocabulary
    vectorize_layer = tf.keras.layers.TextVectorization(vocabulary=vocabulary)

    # Remove stopwords (helper function)
    def remove_stopwords(sentence):
        stopwords = [
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
            "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
            "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
            "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's",
            "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our",
            "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some",
            "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
            "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
            "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when",
            "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would",
            "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
        ]
        sentence = sentence.lower()
        word_list = sentence.split()
        words = [w for w in word_list if w not in stopwords]
        return " ".join(words)

    # Preprocess and predict genre
    processed_text = remove_stopwords(text)
    vectorized_text = vectorize_layer([processed_text])
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(
        vectorized_text, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE
    )
    genre_probabilities = model_lstm.predict(padded_text)
    predicted_genre_index = np.argmax(genre_probabilities, axis=1)[0]
    predicted_genre = label_encoder.inverse_transform([predicted_genre_index])[0]

    return predicted_genre    

## Recomendation System
def predict_book_recommendation(user, filtered_books_df, unique_book_titles, unique_user_ids, model_weights_path, top_n=3):
    # Load and preprocess data
    books = tf.data.Dataset.from_tensor_slices(dict(filtered_books_df[['book_title']]))
    books = books.map(lambda x: x["book_title"])

    # Define the recommendation model
    class BookModel(tfrs.models.Model):
        def __init__(self, rating_weight: float, retrieval_weight: float, unique_book_titles, unique_user_ids) -> None:
            super().__init__()
            embedding_dimension = 32

            self.book_model: tf.keras.layers.Layer = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_book_titles, mask_token=None),
                tf.keras.layers.Embedding(len(unique_book_titles) + 1, embedding_dimension)
            ])
            self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
            ])
            self.rating_model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1),
            ])
            self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()],
            )
            self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=books.batch(248).map(self.book_model)
                )
            )
            self.rating_weight = rating_weight
            self.retrieval_weight = retrieval_weight

        def call(self, features: dict) -> tf.Tensor:
            user_embeddings = self.user_model(features["user_id"])
            book_embeddings = self.book_model(features["book_title"])
            return (
                user_embeddings,
                book_embeddings,
                self.rating_model(tf.concat([user_embeddings, book_embeddings], axis=1)),
            )

        def compute_loss(self, features: dict, training=False) -> tf.Tensor:
            ratings = features.pop("rating")
            user_embeddings, book_embeddings, rating_predictions = self(features)
            rating_loss = self.rating_task(labels=ratings, predictions=rating_predictions)
            retrieval_loss = self.retrieval_task(user_embeddings, book_embeddings)
            return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)

    # Initialize the model
    model = BookModel(rating_weight=1.0, retrieval_weight=1.0,
                      unique_book_titles=unique_book_titles, unique_user_ids=unique_user_ids)
    
    # Build the model with dummy data
    dummy_features = {
        "user_id": tf.constant([unique_user_ids[0]]),
        "book_title": tf.constant([unique_book_titles[0]]),
        "rating": tf.constant([1.0]),
    }
    _ = model(dummy_features)  # This builds the model

    # Load pre-trained weights
    model.load_weights(model_weights_path)

    # Build index for retrieval
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        tf.data.Dataset.zip((books.batch(100), books.batch(100).map(model.book_model)))
    )

    # Generate recommendations
    _, titles = index(tf.constant([str(user)]))
    recommended_titles = [title.decode("utf-8") for title in titles[0, :top_n].numpy()]

    # Filter and return details of the recommended books
    recommendations = filtered_books_df[filtered_books_df['book_title'].isin(recommended_titles)]
    return recommendations.to_dict(orient="records")

"""## OCR"""

def extract_text_from_image(image_path):
        # Open the image
        img = Image.open(image_path)
        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(img)

        # Clean the extracted text
        preprocessed_text = ' '.join(extracted_text.split())

        # Display the extracted text
        print("Extracted text from the image:")
        print(preprocessed_text)
        return preprocessed_text
# Function to recommend books based on user input
def recommend_books(user_ids_df, content_df, predicted_genre, top_n=3, choice="only_genre"):
    user = random.choice(user_ids_df['user_id'].tolist())
    if choice == "only_genre":
        # Filter books only by the predicted genre
        content_df = content_df[content_df['genre'] == predicted_genre].drop_duplicates()
        predict_book_recomendation(user, content_df, top_n=3)
    elif choice == "all_genre":
        # Recommend books from all genres, including the predicted one
        content_df = content_df.drop_duplicates()
        predict_book_recomendation(user, content_df, top_n=3)
    elif choice == "no_genre":
        print("No recommendations will be provided. Thank you!")
    else:
        print("Invalid choice. No recommendations provided.")

# Full workflow to extract text, predict genre, and recommend books
def extract_predict_recommend(title_ocr, user_ids_df, content_df, top_n=3, choice="only_genre"):
    predicted_genre = predict_genre_book(title_ocr)
    if predicted_genre:
      print(f"\nPredicted Genre: {predicted_genre}")
      recommend_books(user_ids_df, content_df, predicted_genre, top_n, choice)
    else:
      print("Failed to predict the genre.")
