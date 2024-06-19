import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from CSV file
data = pd.read_csv('data.csv')

# Check the structure of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Check the distribution of sentiments
label_counts = data['sentiment'].value_counts()
print(label_counts)

# Plot the distribution of sentiments
sns.countplot(x='sentiment', data=data)
plt.show()

# Features and target variable
X = data['text']
y = data['sentiment']

# Convert text data to numerical data using TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
