# Book Recommendation System

A content-based book recommendation system using Natural Language Processing (NLP) and Machine Learning to deliver personalized book suggestions based on user preferences.

## 📚 Project Overview

This project develops an intelligent book recommendation system that analyzes book descriptions, genres, authors, and ratings to provide tailored recommendations. The system processes a dataset of 100,000 books and uses TF-IDF vectorization, Logistic Regression for genre classification, and cosine similarity for generating recommendations.

### Key Features

- **Content-Based Filtering**: Recommends books based on book attributes (descriptions, genres, authors)
- **Genre Classification**: Automatically predicts missing genres with 85% accuracy
- **Personalized Recommendations**: Supports user profile-based filtering by:
  - Preferred genres
  - Favorite authors
  - Minimum rating threshold
  - Previously liked books
- **High Precision**: Achieves 100% precision and satisfaction in recommendations
- **Diversity Enhancement**: Includes soft author filtering and diversity penalties to ensure varied recommendations

## 🛠️ Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computations
  - `scikit-learn` - Machine learning algorithms (Logistic Regression, TF-IDF)
  - `nltk` - Natural language processing (tokenization, lemmatization, stopword removal)
  - `matplotlib` - Data visualization
  - `pickle` - Model serialization

## 📊 Dataset

- **Source**: Augmented Goodreads dataset
- **Size**: 100,000 book records
- **Attributes**:
  - Book ID
  - Title
  - Authors
  - Genre
  - Description
  - Average Ratings (1-5 scale)

## 🚀 Installation

### Prerequisites

1. Install Python 3.x from [python.org](https://www.python.org/downloads/)

2. Install required libraries:
```bash
pip install pandas numpy scikit-learn nltk matplotlib
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📁 Project Structure

```
book-recommendation-system/
│
├── phase1.py                              # Data cleaning script
├── phase2.py                              # Preprocessing and genre classification
├── phase3.py                              # Genre prediction for missing values
├── phase4.py                              # Recommendation system implementation
│
├── cleanedbooks.csv                       # Main dataset
│
├── models/
│   ├── logistics_regression_classifier.pkl   # Trained genre classifier
│   └── tfidf_vectorizer.pkl                   # TF-IDF vectorizer
│
├── evaluation_metrics.png                 # Performance visualization
└── README.md                              # Project documentation
```

## 📖 Usage

### Phase 1: Data Cleaning
```bash
python phase1.py
```
- Removes duplicate entries
- Imputes missing ratings with mean values
- Fills missing descriptions with placeholders
- Marks missing genres for prediction

### Phase 2: Preprocessing and Genre Classification
```bash
python phase2.py
```
- Preprocesses book descriptions (tokenization, lemmatization, stopword removal)
- Trains Logistic Regression classifier for genre prediction
- Saves trained model and TF-IDF vectorizer

### Phase 3: Genre Prediction
```bash
python phase3.py
```
- Predicts missing genres using the trained classifier
- Updates the dataset with predicted genres

### Phase 4: Get Recommendations
```bash
python phase4.py
```

#### Book-Based Recommendations
```python
# Get recommendations similar to a specific book
recommendations = recommend_books(book_id=5, top_n=10)
print(recommendations)
```

#### User Profile-Based Recommendations
```python
# Define user preferences
user_profile = {
    'preferred_genres': ['fantasy', 'mystery'],
    'min_rating': 4.0,
    'preferred_authors': ['J.K. Rowling', 'Patrick Rothfuss'],
    'liked_books': [5, 12346]
}

# Get personalized recommendations
recommendations = recommend_books(user_profile=user_profile, top_n=50)
print(recommendations)
```

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| **Precision** | 1.00 (100%) |
| **Recall** | 0.95 (95%) |
| **Satisfaction** | 1.00 (100%) |
| **Genre Classification Accuracy** | 85% |

### Metric Definitions

- **Precision**: Fraction of recommended books that are relevant to the user
- **Recall**: Fraction of all relevant books that were recommended
- **Satisfaction**: Fraction of recommendations meeting the rating threshold

**Note**: Low recall is due to the large candidate pool (17,020+ relevant books), but high precision ensures all recommendations are relevant.

## 🎯 Algorithm Details

### Text Preprocessing
1. Lowercase conversion
2. Tokenization
3. Stopword removal
4. Lemmatization
5. TF-IDF vectorization

### Genre Classification
- **Algorithm**: Logistic Regression
- **Training Split**: 80% train, 20% test
- **Accuracy**: 85%-90%

### Recommendation Engine
- **Technique**: Content-Based Filtering
- **Similarity Measure**: Cosine Similarity
- **Scoring**: `score = similarity × average_rating`

## 🔮 Future Enhancements

1. **Hybrid Filtering**: Integrate collaborative filtering to improve recall
2. **Advanced NLP**: Use BERT or transformer models for better text representations
3. **Dataset Deduplication**: Remove redundant entries to improve diversity
4. **Web Application**: Deploy as a Flask/Django web app with user interface
5. **User Feedback Loop**: Collect ratings to refine recommendations
6. **Multi-language Support**: Extend to books in multiple languages

## 🏫 Institution

**Virtual University of Pakistan**  
Department of Computer Sciences  
Software Projects & Research Section

## 📄 License

This project is submitted as part of BS Computer Science degree requirements at Virtual University of Pakistan.

## 📚 References

- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *Computer*, 42(8), 30–37.
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781.
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

**Project Status**: ✅ Completed (2025)

**Last Updated**: December 2025
