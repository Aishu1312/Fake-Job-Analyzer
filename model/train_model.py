import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("../data/dataset.csv")

# Keep required columns
df = df[['description', 'fraudulent']].dropna()

# Features & Labels
X = df['description']
y = df['fraudulent']

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ Model & Vectorizer saved successfully")
