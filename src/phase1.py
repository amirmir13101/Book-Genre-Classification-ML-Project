import pandas as pd
import numpy as np
import os

# Define the required attributes as per the document
required_attributes = [
    "genre", "Book ID", "Title", "Authors", "Genre", "Category", "Description",
    "Average ratings", "ISBN", "Language", "Number of pages", "Rating counts",
    "Text Review Counts", "Price", "Publication Date", "Publisher"
]

# Function to load and preprocess dataset
def prepare_dataset(input_file, output_file):
    try:
        # Step 1: Load the dataset 
        print("Loading dataset...")
        df = pd.read_csv(input_file, encoding='utf-8', on_bad_lines='skip')

        # Strip any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        
        # Step 2: Inspect available columns
        print("Available columns in dataset:", list(df.columns))

        # Step 3: Rename columns to match required attributes (if possible)
        column_mapping = {
            'book_id': 'Book ID',
            'title': 'Title',
            'authors': 'Authors',
            'average_rating': 'Average ratings',
            'isbn': 'ISBN',
            'language_code': 'Language',
            'num_pages': 'Number of pages',
            'ratings_count': 'Rating counts',
            'text_reviews_count': 'Text Review Counts',
            'publication_date': 'Publication Date',
            'publisher': 'Publisher'
        }
        
        # Renaming columns
        df = df.rename(columns=column_mapping)

        # Step 4: Check for missing required attributes and add them with placeholders
        for attr in required_attributes:
            if attr not in df.columns:
                print(f"Attribute '{attr}' not found. Adding with placeholder values.")
                if attr == "Price":
                    df[attr] = np.random.uniform(5.0, 50.0, size=len(df))  # Random prices between 5 and 50
                elif attr == "Description":
                    df[attr] = "No description available"  # Placeholder text
                elif attr == "Category" or attr == "genre" or attr == "Genre":
                    df[attr] = "Unknown"  # Placeholder for genre/category
                else:
                    df[attr] = np.nan  # NaN for other missing numeric fields

        # Step 5: Ensure 'Genre' is present (combining if duplicated due to case sensitivity)
        if 'genre' in df.columns and 'Genre' in df.columns:
            df['Genre'] = df['Genre'].combine_first(df['genre'])
            df = df.drop(columns=['genre'])

        # Step 6: Basic data cleaning
        df = df.drop_duplicates(subset=['Book ID', 'Title'], keep='first')
        df = df.fillna({'Description': 'No description available', 'Genre': 'Unknown'})

        # Step 7: Remove any empty columns
        df = df.dropna(axis=1, how='all')

        # Step 8: Save the processed dataset
        df.to_csv(output_file, index=False)
        print(f" Dataset cleaned and saved as '{output_file}'")
        print(" Final columns:", list(df.columns))
        print(f" Number of records: {len(df)}")
        print(" Data is cleaned in 'cleanedbooks.csv'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Please ensure it exists.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Main execution
if __name__ == "__main__":
    input_file = "augmented_books_100000.csv"  # Assuming this is the Goodreads dataset file name
    output_file = "cleanedbooks.csv"  # Output file name changed to 'cleanedbooks.csv'
    
    if os.path.exists(input_file):
        prepare_dataset(input_file, output_file)
    else:
        print(f"Please download the dataset (e.g., from Kaggle) and place it as '{input_file}' in the current directory.")
        print("Suggested source: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks")