import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline

# Load cleaned dataset
df = pd.read_csv('cleaned_environmental_news.csv')

# Topic Modeling with LDA
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(tfidf_matrix)

# Display top words in each topic
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))

tf_feature_names = vectorizer.get_feature_names_out()
display_topics(lda, tf_feature_names, 10)

# Sentiment Analysis with BERT
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment(text):
    result = sentiment_pipeline(text[:512])
    return result[0]['label']

df['sentiment'] = df['Article Text'].apply(get_sentiment)

# Save the results
df.to_csv('environment_news_with_topics_and_sentiment.csv', index=False)
