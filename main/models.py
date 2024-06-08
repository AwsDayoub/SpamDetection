import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from django.db import models

class SpamDetector(models.Model):
    @staticmethod
    def predict(email_text):
        data = pd.read(r'C:\Users\Aws\Desktop\projects\Spam Detection\data\mail_data.csv')

        emails = data['Message'].values
        labels = data['Category'].values

        stemmer = PorterStemmer()

        # Define a function to preprocess the text
        def preprocess(text):
            # Lowercase the text
            text = text.lower()
            # Remove non-alphabetic characters
            text = re.sub('[^a-z]', ' ', text)
            # Tokenize the text
            words = text.split()
            # Remove stopwords and stem the words
            words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
            # Join the words back into a string
            text = ' '.join(words)
            return text

        # Preprocess all emails
        emails = [preprocess(email) for email in emails]

        # Convert emails to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(emails)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)