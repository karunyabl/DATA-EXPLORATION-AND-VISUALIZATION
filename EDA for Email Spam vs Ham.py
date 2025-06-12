# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string

# Step 2: Load Dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Understand Dataset Structure
print(df.head())
print(df.info())
print(df.describe())

# Step 4: Check for Missing Values
print(df.isnull().sum())

# Step 5: Analyze Class Distribution
sns.countplot(data=df, x='label')
plt.title('Spam vs Ham Count')
plt.show()

# Step 6: Visualize Data Distribution
df['message_length'] = df['message'].apply(len)
sns.histplot(data=df, x='message_length', hue='label', bins=50)
plt.title('Message Length Distribution')
plt.show()

# Step 7: Text-specific Analysis

# Function to clean text
def clean_text(text):
    text = text.lower()
    return ''.join([char for char in text if char not in string.punctuation])

df['clean_message'] = df['message'].apply(clean_text)

# Word Frequency
from collections import Counter

spam_words = ' '.join(df[df['label'] == 'spam']['clean_message']).split()
ham_words = ' '.join(df[df['label'] == 'ham']['clean_message']).split()

spam_freq = Counter(spam_words).most_common(20)
ham_freq = Counter(ham_words).most_common(20)

# Word Clouds
spam_wc = WordCloud(width=600, height=400).generate(' '.join(spam_words))
ham_wc = WordCloud(width=600, height=400).generate(' '.join(ham_words))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(spam_wc, interpolation='bilinear')
plt.title('Spam Word Cloud')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(ham_wc, interpolation='bilinear')
plt.title('Ham Word Cloud')
plt.axis('off')

plt.show()

# Step 8: Correlations (Only relevant if you extract numeric features, optional)
# Not much to correlate here unless you engineer more features.

# Step 9: Insights
print("Average message length (spam):", df[df['label'] == 'spam']['message_length'].mean())
print("Average message length (ham):", df[df['label'] == 'ham']['message_length'].mean())
