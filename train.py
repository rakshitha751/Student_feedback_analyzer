from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Sample data
texts = [
    "lab computers are slow",
    "teacher explains well",
    "class is boring",
    "need more examples"
]

labels = ["Negative", "Positive", "Negative", "Neutral"]

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


# Model (IMPORTANT CHANGE)
model = LogisticRegression()
model.fit(X, y)

# Save
# Save files
pickle.dump(model, open("new_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("Model trained and saved successfully")