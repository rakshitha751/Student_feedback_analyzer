import streamlit as st
import pickle
import spacy

# =========================
# Load spaCy Model
# =========================

nlp = spacy.load("en_core_web_md")

# =========================
# Load Saved Models
# =========================

sentiment_model = pickle.load(
    open("sentiment_model.pkl", "rb")
)

sentiment_encoder = pickle.load(
    open("sentiment_encoder.pkl", "rb")
)

category_model = pickle.load(
    open("category_model.pkl", "rb")
)

category_encoder = pickle.load(
    open("category_encoder.pkl", "rb")
)

# =========================
# Text Cleaning Function
# =========================

def clean_text(text):

    doc = nlp(text.lower())

    tokens = []

    for token in doc:

        if not token.is_stop and not token.is_punct:

            tokens.append(token.lemma_)

    return " ".join(tokens)

# =========================
# Issue Mapping
# =========================

issue_map = {

    "computer": "Computer Performance",
    "computers": "Computer Performance",
    "system": "Computer Performance",
    "systems": "Computer Performance",
    "lab": "Computer Performance",

    "wifi": "Internet Connectivity",
    "internet": "Internet Connectivity",
    "network": "Internet Connectivity",

    "faculty": "Teaching Quality",
    "teacher": "Teaching Quality",
    "teaching": "Teaching Quality",

    "library": "Library Resources",
    "book": "Library Resources",
    "books": "Library Resources",

    "placement": "Placement Support",
    "training": "Placement Support",
    "interview": "Placement Support",

    "hostel": "Hostel Facilities",

    "canteen": "Food Quality",
    "food": "Food Quality",
    "meal": "Food Quality",

    "sports": "Sports Facilities",

    "bus": "Transport Services",
    "transport": "Transport Services"
}

# =========================
# Suggestion Mapping
# =========================

suggestion_map = {

    "Computer Performance":
        "Upgrade Lab Systems",

    "Internet Connectivity":
        "Improve WiFi Infrastructure",

    "Teaching Quality":
        "Provide More Interactive Teaching",

    "Library Resources":
        "Add More Updated Books",

    "Placement Support":
        "Increase Industry Collaborations",

    "Hostel Facilities":
        "Improve Hostel Maintenance",

    "Food Quality":
        "Improve Canteen Food Quality",

    "Sports Facilities":
        "Upgrade Sports Equipment",

    "Transport Services":
        "Increase Bus Availability",

    "General Feedback":
        "Review Feedback Manually"
}

# =========================
# Issue Extraction
# =========================

def extract_issue(text):

    text = text.lower()

    for keyword, issue in issue_map.items():

        if keyword in text:

            return issue

    return "General Feedback"

# =========================
# Suggestion Generator
# =========================

def get_suggestion(issue):

    return suggestion_map.get(
        issue,
        "No Suggestion Available"
    )

# =========================
# Main Analysis Function
# =========================

def analyze_feedback(feedback):

    cleaned = clean_text(feedback)

    vector = nlp(cleaned).vector.reshape(1, -1)

    sentiment = sentiment_encoder.inverse_transform(
        sentiment_model.predict(vector)
    )[0]

    category = category_encoder.inverse_transform(
        category_model.predict(vector)
    )[0]

    issue = extract_issue(feedback)

    suggestion = get_suggestion(issue)

    return category, sentiment, issue, suggestion

# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Student Feedback Analyzer",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Feedback Analyzer")

st.write(
    "Enter student feedback and analyze the category, sentiment, issue, and suggestion."
)

feedback = st.text_area(
    "Enter Student Feedback",
    height=150
)

if st.button("Analyze Feedback"):

    if feedback.strip() == "":

        st.warning("Please enter feedback.")

    else:

        category, sentiment, issue, suggestion = analyze_feedback(
            feedback
        )

        st.success("Analysis Complete")

        st.subheader("Results")

        st.write(f"**Category:** {category}")

        st.write(f"**Sentiment:** {sentiment}")

        st.write(f"**Issue:** {issue}")

        st.write(f"**Suggestion:** {suggestion}")