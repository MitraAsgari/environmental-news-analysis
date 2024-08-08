import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Data Preprocessing Function
def preprocess_text(text):
    # Tokenize and remove stopwords
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

# Load dataset
df = pd.read_csv('/content/guardian_environment_news.csv')

# Print column names to check for the correct text column
print(df.columns.tolist())

# Assuming the text column is named 'Article Text'
text_column = 'Article Text'

# Remove NaN values and convert non-string values to strings
df[text_column] = df[text_column].fillna('').astype(str)

# Apply preprocessing
df['cleaned_text'] = df[text_column].apply(preprocess_text)

# Save the cleaned dataset for further analysis
df.to_csv('cleaned_environmental_news.csv', index=False)
