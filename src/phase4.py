import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import nltk
from phase3 import preprocess_text, predict_genre  # Import from phase3.py

# Ensure NLTK data is available
nltk_data_required = ['punkt', 'stopwords', 'wordnet']
for data in nltk_data_required:
    try:
        nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else f'corpora/{data}')
    except LookupError:
        print(f"Downloading NLTK {data}...")
        nltk.download(data, quiet=True)
print("✅ NLTK data verified")

# Load dataset
df = pd.read_csv('cleanedbooks.csv')

# Load saved model and vectorizer
try:
    classifier = joblib.load('logistics_regression_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ Model and vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")
    exit(1)

# Debug: Print unique genres and author diversity
print("Unique genres in dataset:", sorted(df['Genre'].str.lower().unique()))
print("Number of unique authors in fantasy/mystery:", len(df[df['Genre'].str.lower().isin(['fantasy', 'mystery'])]['Authors'].unique()))

# Predict genres for missing or unknown values
df['Genre'] = df.apply(
    lambda row: predict_genre(row['Description'])[0] if pd.isna(row['Genre']) or row['Genre'].lower() == 'unknown' else row['Genre'],
    axis=1
)

# Preprocess all descriptions
df['Processed_Description'] = df['Description'].apply(preprocess_text)

# Generate TF-IDF vectors for all books
X = vectorizer.transform(df['Processed_Description']).toarray()

def recommend_books(book_id=None, user_profile=None, top_n=5):
    """
    Recommend books based on a book ID or user profile.
    
    Args:
        book_id (int): ID of a book to find similar books (optional).
        user_profile (dict): User preferences (preferred_genres, preferred_authors, min_rating, liked_books).
        top_n (int): Number of recommendations to return.
    
    Returns:
        DataFrame or str: Top N recommended books or error message.
    """
    # Initialize candidates as all books
    candidates = df.copy()
    candidates['similarity'] = 1.0  # Default similarity score

    # Handle book_id-based recommendations
    if book_id is not None:
        book = df[df['Book ID'] == book_id]
        if book.empty:
            return "Book not found."
        
        # Get genre (predict if missing)
        genre = book['Genre'].iloc[0]
        if pd.isna(genre) or genre.lower() == 'unknown':
            genre, _ = predict_genre(book['Description'].iloc[0])
        
        # Filter candidates by genre (case-insensitive)
        candidates = candidates[candidates['Genre'].str.lower() == genre.lower()]
        print(f"After genre filter (book_id={book_id}): {len(candidates)} candidates")
        
        # Compute similarity
        book_idx = df.index[df['Book ID'] == book_id].tolist()[0]
        candidate_indices = candidates.index.tolist()
        similarities = cosine_similarity(X[book_idx:book_idx+1], X[candidate_indices]).flatten()
        candidates['similarity'] = similarities

    # Apply user profile filters
    if user_profile:
        # Convert user profile genres to lowercase
        if 'preferred_genres' in user_profile and user_profile['preferred_genres']:
            preferred_genres = [g.lower() for g in user_profile['preferred_genres']]
            candidates = candidates[candidates['Genre'].str.lower().isin(preferred_genres)]
            print(f"After genre filter (user_profile): {len(candidates)} candidates")
        if 'preferred_authors' in user_profile and user_profile['preferred_authors']:
            # Soft author filtering
            author_pattern = '|'.join(user_profile['preferred_authors']).replace('.', r'\.')
            candidates['is_preferred_author'] = candidates['Authors'].str.contains(author_pattern, case=False, na=False, regex=True)
            print(f"After author filter (soft): {len(candidates)} candidates")
        if 'min_rating' in user_profile:
            candidates = candidates[candidates['Average ratings'] >= user_profile['min_rating']]
            print(f"After rating filter: {len(candidates)} candidates")
        if 'liked_books' in user_profile and user_profile['liked_books']:
            # Validate liked_books
            valid_liked_books = [bid for bid in user_profile['liked_books'] if bid in df['Book ID'].values]
            print(f"Valid liked books: {valid_liked_books}")
            if not valid_liked_books:
                print("No valid liked books found in dataset.")
            else:
                liked_indices = df.index[df['Book ID'].isin(valid_liked_books)].tolist()
                if liked_indices and not candidates.empty:
                    liked_similarities = cosine_similarity(X[liked_indices], X[candidates.index]).mean(axis=0)
                    # Balanced liked books similarity
                    candidates['similarity'] = candidates['similarity'] * (0.7 + 0.3 * liked_similarities)
                else:
                    print("Skipping liked books similarity due to empty candidates or no valid liked books.")

    # If no candidates remain, return message with fallback suggestion
    if candidates.empty:
        return f"No recommendations found matching your preferences. Try relaxing filters (e.g., genres: {', '.join(df['Genre'].str.lower().unique()[:5])})."

    # Random sampling for large candidate pools (max 5000 to improve diversity)
    if len(candidates) > 5000:
        candidates = candidates.sample(n=min(5000, len(candidates)), random_state=42)
        print(f"After random sampling: {len(candidates)} candidates")

    # Rank by weighted score (similarity * rating, small boost for preferred authors)
    candidates['score'] = candidates['similarity'] * candidates['Average ratings']
    if 'is_preferred_author' in candidates:
        candidates['score'] = candidates['score'] * (1 + 0.1 * candidates['is_preferred_author'].astype(int))
    
    # Add diversity factor (penalize duplicate authors/titles)
    candidates['author_count'] = candidates.groupby('Authors')['Authors'].transform('count')
    candidates['title_count'] = candidates.groupby('Title')['Title'].transform('count')
    candidates['diversity_factor'] = 1 / (1 + 0.3 * (candidates['author_count'] + candidates['title_count']))
    candidates['score'] = candidates['score'] * candidates['diversity_factor']

    # Sort and return top N
    recommendations = candidates.sort_values(by='score', ascending=False)
    print(f"Unique authors in recommendations: {len(recommendations['Authors'].unique())}")
    return recommendations[['Book ID', 'Title', 'Authors', 'Genre', 'Average ratings', 'similarity']].head(top_n)

def evaluate_recommendations(user_profile, top_n=50):
    """
    Evaluate the recommendation system using precision, recall, and satisfaction.
    
    Args:
        user_profile (dict): User preferences for evaluation.
        top_n (int): Number of recommendations to evaluate.
    
    Returns:
        dict: Precision, recall, and satisfaction metrics.
    """
    # Simulate ground-truth: books with rating >= user_profile['min_rating'] in preferred genres
    preferred_genres = [g.lower() for g in user_profile['preferred_genres']]
    relevant_books = df[df['Genre'].str.lower().isin(preferred_genres) & 
                       (df['Average ratings'] >= user_profile['min_rating'])]
    print(f"Total relevant books: {len(relevant_books)}")
    recs = recommend_books(user_profile=user_profile, top_n=top_n)

    if isinstance(recs, str):  # No recommendations found
        return {'precision': 0.0, 'recall': 0.0, 'satisfaction': 0.0}

    # Compute precision and recall
    y_true = df['Book ID'].isin(relevant_books['Book ID']).astype(int)
    y_pred = df['Book ID'].isin(recs['Book ID']).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    # Compute satisfaction (fraction of recommendations with rating >= min_rating)
    satisfaction = (recs['Average ratings'] >= user_profile['min_rating']).mean()

    return {'precision': precision, 'recall': recall, 'satisfaction': satisfaction}

def plot_evaluation_metrics(metrics):
    """
    Plot evaluation metrics and save as PNG.
    
    Args:
        metrics (dict): Precision, recall, and satisfaction.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange'])
    plt.title('Recommendation System Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    plt.savefig('evaluation_metrics.png')
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example 1: Recommend based on a book (e.g., Book ID 5)
    print("\n=== Book-Based Recommendations (Book ID 5) ===")
    book_recs = recommend_books(book_id=5, top_n=3)
    print(book_recs)
    print("-" * 50)

    # Example 2: Recommend based on user profile
    user_profile = {
        'preferred_genres': ['fantasy', 'mystery'],
        'min_rating': 4.0,
        'preferred_authors': ['J.K. Rowling'],
        'liked_books': [5]
    }
    print("\n=== User Profile-Based Recommendations ===")
    user_recs = recommend_books(user_profile=user_profile, top_n=3)
    print(user_recs)
    print("-" * 50)

    # Evaluate recommendations
    print("\n=== Evaluation Metrics ===")
    metrics = evaluate_recommendations(user_profile, top_n=50)
    print(f"Precision: {metrics['precision']:.2f} (fraction of recommended books that are relevant)")
    print(f"Recall: {metrics['recall']:.2f} (fraction of relevant books recommended)")
    print(f"Satisfaction: {metrics['satisfaction']:.2f} (fraction of recommendations meeting rating threshold)")
    print("Plot saved as 'evaluation_metrics.png'")

    # Plot metrics
    plot_evaluation_metrics(metrics)