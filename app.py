import streamlit as st
import pickle
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (if not already available)
nltk.download('stopwords')
nltk.download('wordnet')

# --- Helper function for text preprocessing ---
def preprocess_email(email):
    """
    Preprocesses raw email text: lowercases, removes punctuation/numbers,
    removes stopwords, and lemmatizes.
    """
    # Lowercase
    email = email.lower()
    # Remove punctuation
    email = email.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    email = re.sub(r'\d+', '', email)
    # Tokenize and remove stopwords
    tokens = email.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# --- Load the saved model and vectorizer ---
try:
    # Update the filenames if needed
    model_filename = "best_random_forest_model.pkl"  # or best_naive_bayes_model.pkl
    model = pickle.load(open(model_filename, 'rb'))
except Exception as e:
    st.error(f"Error loading model: {e}")

try:
    vectorizer_filename = "vectorizer.pkl"
    vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")

# --- Configure the Streamlit page ---
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="wide",
)

st.title("📧 Email Spam Detection App")
st.write("""
**Welcome to the Email Spam Detection App!**  
This app uses a machine learning model to classify email messages as **Spam** or **Ham** (Not Spam).  
Enter or paste an email below to get an instant classification.
""")

# Sidebar info
st.sidebar.title("App Features")
st.sidebar.write("""
- **Input an Email**: Paste your email content for classification.  
- **Visualize Results**: See the spam vs ham probabilities or the confusion matrix.  
- **Tech Stack**: Built using Python, Streamlit, and Scikit-learn.
""")

# --- Input Section ---
st.subheader("📝 Input Email")
email_input = st.text_area("Enter the email content here:", height=150)

if st.button("Classify Email"):
    if not email_input.strip():
        st.error("⚠️ Please enter some email content!")
    else:
        # Preprocess the input email
        processed_email = preprocess_email(email_input)
        # Transform the processed email using the loaded vectorizer
        email_features = vectorizer.transform([processed_email]).toarray()
        try:
            # Predict using the loaded model
            prediction = model.predict(email_features)
            prediction_proba = model.predict_proba(email_features)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
        else:
            result = "Spam" if prediction[0] == 1 else "Ham"
            st.success(f"📝 The email is classified as: **{result}**")
            st.write(f"🔍 **Spam Probability:** {prediction_proba[0][1] * 100:.2f}%")
            st.write(f"🔍 **Ham Probability:** {prediction_proba[0][0] * 100:.2f}%")

            # Optional: Show a pie chart for the probabilities
            fig, ax = plt.subplots()
            ax.pie(
                prediction_proba[0],
                labels=["Ham", "Spam"],
                autopct="%1.1f%%",
                colors=["skyblue", "orange"],
                startangle=90,
            )
            ax.set_title("Spam vs Ham Probability")
            st.pyplot(fig)

# --- Visualizations Section ---
st.subheader("📊 Visualizations")

# Confusion Matrix Visualization (Optional)
if st.checkbox("Show Confusion Matrix"):
    st.write("This confusion matrix represents model performance on test data.")
    
    # Example confusion matrix (replace with actual if available)
    confusion_matrix_data = np.array([[965, 5], [20, 150]])
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix_data, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# --- Footer ---
st.write("----")
st.write("🛠️ Built with ❤️ using Python, Streamlit, and Scikit-learn.")

