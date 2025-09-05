import streamlit as st
import joblib # To load the model and vectorizer
import os # For checking file existence
import pandas as pd  # Add at the top if not already imported

# --- File Paths for your saved model and vectorizer ---
MODEL_PATH = 'spam_detector_model1.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer1.pkl'

# --- Page Configuration ---
st.set_page_config(
    page_title="Spam email detector",
     page_icon="📧",
    layout="wide"
)


# --- Load the Model and Vectorizer (cached for efficiency) ---
# Use st.cache_resource to load these heavy objects only once when the app starts
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the correct directory.")
        st.stop() # Stop the app if essential files are missing
    if not os.path.exists(VECTORIZER_PATH):
        st.error(f"Error: Vectorizer file '{VECTORIZER_PATH}' not found. Please ensure it's in the correct directory.")
        st.stop() # Stop the app if essential files are missing

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

# --- Streamlit UI ---
st.title("📧Email Spam Classifier")
st.write("This application classifies emails as spam or not spam using a pre-trained Logistic Regression model.")

# Load the pre-trained model and vectorizer
model, vectorizer = load_artifacts()

st.header("Predict if an Email is Spam")

user_input = st.text_area("Enter the email text here:", "Subject: Urgent - Your Amazon account requires verification! Click here immediately.")

if st.button("Classify Email"):
    if user_input:
        # Transform the user input using the loaded vectorizer
        user_input_tfidf = vectorizer.transform([user_input])

        # Make a prediction using the loaded model
        prediction = model.predict(user_input_tfidf)
        prediction_proba = model.predict_proba(user_input_tfidf)

        st.subheader("Prediction Result:")
        # Assuming '1' is spam, '0' is not spam
        if prediction[0] == '1':
            st.error("This email is likely **SPAM**! 🚨")
        else:
            st.success("This email is likely **NOT SPAM**! ✅")
        
        st.write(f"Probability of being 'not spam' (0): {prediction_proba[0][0]:.4f}")
        st.write(f"Probability of being 'spam' (1): {prediction_proba[0][1]:.4f}")

        # Display probabilities as a bar chart
        
        proba_df = pd.DataFrame({
            'Class': ['Not Spam', 'Spam'],
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(proba_df.set_index('Class'))
    else:
        st.warning("Please enter some email text to classify.")



# st.markdown("---")
# st.subheader("How it works:")
# st.write("This app uses a pre-trained TF-IDF vectorizer to transform email text and a pre-trained Logistic Regression model to predict if it's spam.")
# st.write(f"Model loaded from: `{MODEL_PATH}`")
# st.write(f"Vectorizer loaded from: `{VECTORIZER_PATH}`")

# st.write("Here's an illustration of the prediction process:")
    