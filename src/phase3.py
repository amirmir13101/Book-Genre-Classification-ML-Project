import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Define stopwords and lemmatizer
default_stopwords = set(stopwords.words("english"))
custom_stopwords = default_stopwords - {"murder", "detective", "magic", "alien", "warrior", "king"}
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords]
    return " ".join(tokens)

# Load trained model and vectorizer
try:
    classifier = joblib.load("logistics_regression_classifier.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    print("✅ Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("❌ Error: Model or vectorizer file not found. Please ensure 'naive_bayes_genre_classifier.pkl' and 'tfidf_vectorizer.pkl' are in the current directory.")
    exit(1)
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {str(e)}")
    exit(1)

def predict_genre(description, threshold=0.5):
    try:
        processed_desc = preprocess_text(description)
        desc_vector = vectorizer.transform([processed_desc]).toarray()
        predicted_genre = classifier.predict(desc_vector)[0]
        probabilities = classifier.predict_proba(desc_vector)[0]
        confidence = np.max(probabilities)

        if confidence < threshold:
            top_two = np.argsort(probabilities)[-2:]
            top_two_genres = [classifier.classes_[i] for i in top_two]
            return f"Uncertain (Top 2: {top_two_genres[1]}: {probabilities[top_two[1]]*100:.2f}%, {top_two_genres[0]}: {probabilities[top_two[0]]*100:.2f}%)"
        
        return predicted_genre, confidence * 100
    except Exception as e:
        return f"Error predicting genre: {str(e)}"

# Test descriptions
test_descriptions = [
    "A young girl discovers she has magical powers and must save her kingdom from an evil sorcerer.",  # Fantasy  
    "A retired detective is pulled back into a case when a serial killer resurfaces after 20 years.",  # Mystery  
    "A scientist accidentally opens a portal to another dimension, leading to an alien invasion.",  # Science Fiction  
    "Two childhood friends reunite in Paris and realize their love for each other after years apart.",  # Romance  
    "In a post-apocalyptic world, a young warrior must fight to protect his people from a ruthless warlord.",  # Dystopian
]

# Main execution
if __name__ == "__main__":
    print("\n🔍 **Genre Predictions with Confidence Scores:**")
    for desc in test_descriptions:
        result = predict_genre(desc, threshold=0.5)
        if isinstance(result, tuple):
            genre, confidence = result
            print(f"📖 **Description:** {desc}")
            print(f"🔹 **Predicted Genre:** {genre}, 🎯 **Confidence:** {confidence:.2f}%")
        else:
            print(f"📖 **Description:** {desc}")
            print(f"🔹 **Result:** {result}")
        print("-" * 50)