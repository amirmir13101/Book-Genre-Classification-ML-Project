import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import joblib  # Import joblib to save the model and vectorizer

try:
    # Timer Start
    start_time = time.time()

    #  Step 1: Load Dataset
    df = pd.read_csv("cleanedbooks.csv")
    print("Dataset loaded successfully")

    # Step 2: Data Preprocessing
    nltk.download("punkt", quiet=True)  # This ensures 'punkt' is available
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    # Custom Stopwords
    default_stopwords = set(stopwords.words("english"))
    custom_stopwords = default_stopwords - {"murder", "detective", "magic", "alien", "warrior", "king"}

    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = str(text).lower()  # Convert to string and lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        tokens = word_tokenize(text)  # Tokenization
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords]  # Lemmatization & Stopword removal
        return " ".join(tokens)

    # Apply Preprocessing with timer
    preprocess_start = time.time()
    df["Processed_Description"] = df["Description"].apply(preprocess_text)
    preprocess_time = time.time() - preprocess_start
    print(f"Preprocessing Time: {preprocess_time:.2f} sec")

    # 📌 Step 3: Convert Text into Numerical Features (TF-IDF)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=5, sublinear_tf=True)
    X = vectorizer.fit_transform(df["Processed_Description"]).toarray()
    y = df["Genre"]
    print("✅ Text vectorization completed")

    # 📌 Step 4: Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 📌 Step 5: Train Model with timer
    train_start = time.time()
    classifier = LogisticRegression(max_iter=2, solver="saga", n_jobs=-1)
    classifier.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"✅ Training Time: {train_time:.2f} sec")

    # Step 6: Evaluate Model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.2f}")

    # Step 7: Display Classification Report
    print("\n🔍 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Step 8: Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.show()

    # ✅ Total Time
    total_time = time.time() - start_time
    print(f"✅ Total Execution Time: {total_time:.2f} sec")

    # 📌 Save the model and vectorizer
    joblib.dump(classifier, 'logistics_regression_classifier.pkl')  # Save the classifier
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save the vectorizer

    print("✅ Model and Vectorizer saved as 'logistics_regression_classifier.pkl' and 'tfidf_vectorizer.pkl'.")

except FileNotFoundError:
    print("❌ Error: 'processed_books_dataset.csv' file not found in the current directory")
except Exception as e:
    print(f"❌ An error occurred: {str(e)}")
