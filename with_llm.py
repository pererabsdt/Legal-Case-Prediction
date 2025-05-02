import re
import json
import nltk
import joblib
import PyPDF2
import numpy as np
import pandas as pd
from mistralai import Mistral, UserMessage
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv
import os

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --------------------------
# Step 1: PDF to Raw Text
# --------------------------
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        return ' '.join(page.extract_text() for page in reader.pages if page.extract_text())

# --------------------------
# Step 2: LLM Output
# --------------------------
def call_llm_real(text, api_key):    
    prompt = f"""
    Extract the following fields from the legal case:
    - decision_type out of ["majority opinion", "per curiam", "plurality opinion", "equally divided", "dismissal - rule 46", "dismissal - other", "dismissal - improvidently granted", "dismissal - moot", "memorandum", "opinion of the court"]
    - disposition out of ["reversed/remanded", "affirmed", "reversed", "vacated/remanded", "reversed in-part/remanded", "none", "reversed in-part", "vacated", "vacated in-part/remanded"]
    - first_party
    - second_party
    - facts (summarized)

    Return a JSON object with these keys. Example format:
    {{
      "decision_type": "majority opinion",
      "disposition": "affirmed",
      "first_party": "Smith",
      "second_party": "United States",
      "facts": "A summary of the case facts goes here."
    }}

    Legal case text:
    {text}
    """
    
    client = Mistral(api_key=api_key)

    messages = [{
        "role": "user",
        "content": prompt
    }]
    
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )
    
    return json.loads(response.choices[0].message.content)

# --------------------------
# Step 3: Preprocessing for NLP
# --------------------------
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]
    if flg_stemm:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
    if flg_lemm:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
    return " ".join(lst_text)

# --------------------------
# Step 4: One-Hot Encoding Setup
# --------------------------
def setup_one_hot_encoders():
    # Define possible values for categorical features
    decision_types = [
        "majority opinion", "per curiam", "plurality opinion",
        "equally divided", "dismissal - rule 46", "dismissal - other",
        "dismissal - improvidently granted", "dismissal - moot", 
        "memorandum", "opinion of the court"
    ]
    
    dispositions = [
        "reversed/remanded", "affirmed", "reversed", "vacated/remanded",
        "reversed in-part/remanded", "none", "reversed in-part",
        "vacated", "vacated in-part/remanded"
    ]
    
    # Create and fit one-hot encoders
    decision_type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    decision_type_encoder.fit(np.array(decision_types).reshape(-1, 1))
    
    disposition_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    disposition_encoder.fit(np.array(dispositions).reshape(-1, 1))
    
    return decision_type_encoder, disposition_encoder

# --------------------------
# Step 5: Main Pipeline
# --------------------------
def main(pdf_path, api_key):
    try:
        # Load trained vectorizer, LDA model, and prediction model
        vectorizer = joblib.load("vectorizer.pkl")
        
        # Load or create LDA model - ideally this should also be saved/loaded from training
        # If you have a saved LDA model, load it with:
        try:
            lda_model = joblib.load("lda_model1.pkl")
            print("Loaded existing LDA model")
        except FileNotFoundError:
            # If no saved LDA model exists, create a new one with the same parameters
            print("LDA model not found, creating a new one with n_components=200")
            lda_model = LatentDirichletAllocation(n_components=200, random_state=0)
            
        # Load the trained prediction model
        model = joblib.load("case_prediction_model_1.pkl")
        
        # Setup one-hot encoders
        decision_type_encoder, disposition_encoder = setup_one_hot_encoders()

        # Step 1: Extract raw text from PDF
        raw_text = extract_text_from_pdf(pdf_path)

        # Step 2: Get structured features from LLM (mocked here)
        llm_response = call_llm_real(raw_text, api_key)
        features = json.loads(llm_response)

        # Step 3: Clean "facts" using your original function
        lst_stopwords = nltk.corpus.stopwords.words("english")
        cleaned_facts = utils_preprocess_text(features["facts"], flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords)

        # Step 4: Vectorize cleaned facts text
        X_text = vectorizer.transform([cleaned_facts])
        print(f"Raw text features shape after vectorization: {X_text.shape}")
        
        # Step 5: Apply LDA transformation to reduce dimensions to 200
        # Note: If this is a new LDA model, it's being fit on a single document which is not ideal.
        # In a real scenario, you should use the same LDA model that was fit during training.
        if hasattr(lda_model, 'components_') and lda_model.components_.shape[0] > 0:
            # LDA model is already fitted
            X_lda = lda_model.transform(X_text)
        else:
            # LDA model needs to be fit first (not ideal, but as a fallback)
            X_lda = lda_model.fit_transform(X_text)
            # Save the fitted model for future use
            joblib.dump(lda_model, "lda_model.pkl")
        
        print(f"LDA transformed features shape: {X_lda.shape}")
        
        # Step 6: One-hot encode categorical features
        decision_type_one_hot = decision_type_encoder.transform([[features["decision_type"].lower()]])
        disposition_one_hot = disposition_encoder.transform([[features["disposition"].lower()]])
        
        print(f"Decision type features shape: {decision_type_one_hot.shape}")
        print(f"Disposition features shape: {disposition_one_hot.shape}")

        # Step 7: Determine how to combine features based on the model's expectations
        # If model expects exactly 200 features (LDA only)
        if model.n_features_in_ == 200:
            print("Model expects exactly 200 features (LDA features only)")
            X_combined = X_lda
        # If model expects 200 + categorical features
        elif model.n_features_in_ == X_lda.shape[1] + decision_type_one_hot.shape[1] + disposition_one_hot.shape[1]:
            print("Model expects LDA features + one-hot encoded categorical features")
            X_combined = np.hstack((X_lda, decision_type_one_hot, disposition_one_hot))
        # If model was trained with original numeric encoding (LDA + 2 numeric features)
        elif model.n_features_in_ == X_lda.shape[1] + 2:
            print("Model expects LDA features + 2 numeric categorical features")
            # Convert one-hot back to numeric for compatibility
            decision_type_map = {
                "majority opinion": 0, "per curiam": 1, "plurality opinion": 2,
                "equally divided": 3, "dismissal - rule 46": 4, "dismissal - other": 5,
                "dismissal - improvidently granted": 6, "dismissal - moot": 7, 
                "memorandum": 8, "opinion of the court": 9
            }
            disposition_map = {
                "reversed/remanded": 0, "affirmed": 1, "reversed": 2, "vacated/remanded": 3,
                "reversed in-part/remanded": 4, "none": 5, "reversed in-part": 6,
                "vacated": 7, "vacated in-part/remanded": 8
            }
            decision_type_num = decision_type_map.get(features["decision_type"].lower(), 0)
            disposition_num = disposition_map.get(features["disposition"].lower(), 0)
            X_combined = np.hstack((X_lda, [[decision_type_num, disposition_num]]))
        else:
            # As a fallback, try to match the expected feature count
            expected_features = model.n_features_in_
            available_features = np.hstack((X_lda, decision_type_one_hot, disposition_one_hot))
            
            if available_features.shape[1] > expected_features:
                print(f"Warning: Too many features. Truncating to {expected_features}")
                X_combined = available_features[:, :expected_features]
            elif available_features.shape[1] < expected_features:
                print(f"Warning: Too few features. Padding to {expected_features}")
                padding = np.zeros((available_features.shape[0], expected_features - available_features.shape[1]))
                X_combined = np.hstack((available_features, padding))
            else:
                X_combined = available_features
        
        print(f"Combined features shape: {X_combined.shape}")
        print(f"Model expects shape with {model.n_features_in_} features")

        # Step 8: Predict
        prediction = model.predict(X_combined)[0]
        prediction_proba = model.predict_proba(X_combined)[0]
        
        print(f"Prediction (1 = First Party Wins, 0 = Loses): {prediction}")
        print(f"Prediction probability: {prediction_proba}")
        
        return {
            "prediction": int(prediction),
            "probability": prediction_proba.tolist(),
            "features": {
                "first_party": features["first_party"],
                "second_party": features["second_party"],
                "decision_type": features["decision_type"],
                "disposition": features["disposition"]
            }
        }
    except Exception as e:
        import traceback
        print(f"Error in prediction pipeline: {e}")
        traceback.print_exc()
        return {"error": str(e)}

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    load_dotenv()  # Load environment variables from .env file
    pdf_path = "example_legal_case.pdf" 
    api_key = os.getenv("api_key") 
    result = main(pdf_path, api_key)
    print(json.dumps(result, indent=2))