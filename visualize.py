import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset with topics and sentiment
df = pd.read_csv('environment_news_with_topics_and_sentiment.csv')

# Visualization of Topics using Word Cloud
def plot_word_cloud(model, feature_names, num_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        wordcloud = WordCloud()
        wordcloud.generate_from_frequencies({feature_names[i]: topic[i] for i in topic.argsort()[:-num_top_words - 1:-1]})
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {topic_idx}')
        plt.axis("off")
        plt.show()

# Initialize vectorizer and LDA model
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(tfidf_matrix)

tf_feature_names = vectorizer.get_feature_names_out()
plot_word_cloud(lda, tf_feature_names)

# Visualization of Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()
