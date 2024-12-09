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

sampled_categories = pd.read_csv('data/sampled_categories.csv')
content_df = pd.read_csv('data/content_df.csv')
user_ids_df = pd.read_csv('data/user_ids.csv')

"""# Fitur 2: Predict Book Genre and Give Recomendation

## Text Classification
"""

# Number of examples to use for training
TRAINING_SIZE = 20000

# Vocabulary size of the tokenizer
VOCAB_SIZE = 10000

# Maximum length of the padded sequences
MAX_LENGTH = 32

# Type of padding
PADDING_TYPE = 'pre'

# Specifies how to truncate the sequences
TRUNC_TYPE = 'post'

# Parameters
EMBEDDING_DIM = 16
LSTM_DIM = 32
DENSE_DIM = 24

# Model definition with LSTM
model_lstm = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM)),
    tf.keras.layers.Dense(DENSE_DIM, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Load the model weights (this assumes the model is already defined)
model_lstm.load_weights('model/model_genre_classification_weights.h5')

with open('model/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def remove_stopwords(sentence):
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

    # Sentence converted to lowercase-only
    sentence = sentence.lower()

    # Get all the comma separated words in a list
    word_list = sentence.split()

    # Keep all the words which are not stopwords
    words = [w for w in word_list if w not in stopwords]

    # Reconstruct sentence after discarding all stopwords
    sentence = " ".join(words)

    return sentence

# Load the saved vocabulary
with open('model/vectorizer_vocab.pkl', 'rb') as file:
    vocabulary = pickle.load(file)

# Recreate the TextVectorization layer using the loaded vocabulary
vectorize_layer = tf.keras.layers.TextVectorization(vocabulary=vocabulary)

def predict_genre_book(text):
    # Preprocess the text input (remove stopwords and vectorize)
    processed_text = remove_stopwords(text)  # Assuming 'remove_stopwords' is defined

    # Apply the text vectorization
    vectorized_text = vectorize_layer([processed_text])  # Apply vectorization

    # Pad the vectorized input to ensure it's of the correct length (MAX_LENGTH)
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(vectorized_text, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)

    # Predict the genre probabilities
    genre_probabilities = model_lstm.predict(padded_text)

    # Get the predicted genre index (class with the highest probability)
    predicted_genre_index = np.argmax(genre_probabilities, axis=1)[0]

    # Map the predicted index to the genre name using the inverse of the LabelEncoder
    predicted_genre = label_encoder.inverse_transform([predicted_genre_index])[0]

    return predicted_genre

"""## Recomendation System"""

sampled_categories = sampled_categories.rename(columns={'review/score': 'user_rating', 'User_id': 'user_id', 'Title': 'book_title'})

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

class BookModel(tfrs.models.Model):
    def __init__(self, rating_weight: float, retrieval_weight: float,
                 unique_book_titles, unique_user_ids) -> None:
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

model = BookModel(rating_weight=1.0, retrieval_weight=1.0,
                  unique_book_titles=unique_book_titles, unique_user_ids=unique_user_ids)

# Build the model with dummy data
dummy_features = {
    "user_id": tf.constant([unique_user_ids[0]]),
    "book_title": tf.constant([unique_book_titles[0]]),
    "rating": tf.constant([1.0]),
}
_ = model(dummy_features)  # This builds the model

model.load_weights('model/model_recomendation_weights.h5')

def predict_book_recomendation(user, filtered_books_df, top_n=3):
    books = tf.data.Dataset.from_tensor_slices(dict(filtered_books_df[['book_title']]))
    books = books.map(lambda x: x["book_title"])

    # Create a model that takes in raw query features
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # Recommends books out of the entire books dataset
    index.index_from_dataset(
        tf.data.Dataset.zip((books.batch(100), books.batch(100).map(model.book_model)))
    )

    # Get recommendations
    _, titles = index(tf.constant([str(user)]))
    recommended_titles = [title.decode("utf-8") for title in titles[0, :top_n].numpy()]

    # Filter details from the input DataFrame
    recommendations = filtered_books_df[filtered_books_df['book_title'].isin(recommended_titles)]
    return print(recommendations.to_json(orient='records', lines=False))
    
def predict_rating(user, book):
    trained_book_embeddings, trained_user_embeddings, predicted_rating = model({
          "user_id": np.array([str(user)]),
          "book_title": np.array([book])
      })
    print("Predicted rating for {}: {}".format(book, predicted_rating.numpy()[0][0]))

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
