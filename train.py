import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Sample Dataset
# -----------------------------
data = {
    "feedback": [
        "good teaching",
        "bad explanation",
        "average class",
        "excellent teaching",
        "poor examples",
        "need more examples",
        "very good teacher",
        "not clear explanation"
    ],
    "sentiment": [
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative",
        "Neutral",
        "Positive",
        "Negative"
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["feedback"])

# -----------------------------
# Label Encoding
# -----------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["sentiment"])

# -----------------------------
# Train Model
# -----------------------------
from sklearn.svm import SVC
model = SVC(probability=True)
model.fit(X, y)

# -----------------------------
# Save Files
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

print("✅ Model trained and saved successfully!")