import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data.csv')

# Check the structure of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Check the distribution of segment labels
label_counts = data['segment_label'].value_counts()
print(label_counts)

# Plot the distribution of segment labels
sns.countplot(x='segment_label', data=data)
plt.show()

# Remove classes with fewer than two samples
data = data[data['segment_label'].isin(label_counts[label_counts >= 2].index)]

# Features and target variable
X = data['text']
y = data['segment_label']

# Convert text data to numerical data using TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

# Ensure test_size is large enough to include all classes
test_size = max(0.2, len(data['segment_label'].unique()) / len(data))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=test_size, random_state=42, stratify=y)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
