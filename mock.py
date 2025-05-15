import re
import json
import nltk
import joblib
import PyPDF2
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import OneHotEncoder

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
# Step 2: Mocked LLM Output
# --------------------------
def call_llm_mock(prompt):
    # This replaces Mistral API call for offline testing
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "first_party": "Sarno",
                        "second_party": "Illinois Crime Investigating Commission",
                        "facts": 'The Illinois Crime Investigating Commission was created to investigate organized crime in Illinois. Albert Sarno and Chris Cardi were police officers. The Commission wanted the officers to testify pursuant to an investigation of a "juice loan" or "loan shark" racket. On February 8, 1968, the presiding judge in the Circuit Court of Cook County entered an order requiring the petitioners to appear before the commission under a grant of immunity pursuant to an Illinois statute. On February 24, 1968, the officers appeared, but they refused to answer any questions pleading their right against self-incrimination under the Fifth Amendment. On March 21, 1968, the Commission filed a motion in the Circuit court, and moved the court to enter an order directing the petitioners to appear before the Commission and answer the questions. The officers filed a motion to dismiss or to strike the foregoing motion. The officers motion was denied, and the court entered an order directing the petitioners to appear before the Commission and to answer the questions. The officers refused to obey this subsequent order, still pleading the Fifth Amendment. The officers were found in contempt of court and sentenced to serve a period of six months in the County Jail. The decision of the trial court was appealed and subsequently affirmed by the Illinois Supreme Court. The officers appealed to the Supreme Court.',
                        "decision_type": "per curiam",
                        "disposition": ""
                    })
                }
            }
        ]
    }
    return mock_response["choices"][0]["message"]["content"]

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
        "equally divided", "dismissal - other",
        "dismissal - improvidently granted", "dismissal - moot", 
        "opinion of the court"
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
def main(pdf_path):
    try:
        # Load trained vectorizer, LDA model, and prediction model
        vectorizer = joblib.load("vectorizer.pkl")
        
        # If you have a saved LDA model, load it with:
        try:
            lda_model = joblib.load("lda_model.pkl")
            print("Loaded existing LDA model")
        except FileNotFoundError:
            # If no saved LDA model exists, create a new one with the same parameters
            print("LDA model not found, creating a new one with n_components=200")
            lda_model = LatentDirichletAllocation(n_components=200, random_state=0)
            
        # Load the trained prediction model
        model = joblib.load("case_prediction_model.pkl")
        
        # Setup one-hot encoders
        decision_type_encoder, disposition_encoder = setup_one_hot_encoders()

        # Step 1: Extract raw text from PDF
        raw_text = extract_text_from_pdf(pdf_path)

        # Step 2: Get structured features from LLM (mocked here)
        llm_response = call_llm_mock(raw_text)
        features = json.loads(llm_response)

        # Step 3: Clean "facts" using your original function
        lst_stopwords = nltk.corpus.stopwords.words("english")
        cleaned_facts = utils_preprocess_text(features["facts"], flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords)

        # Step 4: Vectorize cleaned facts text
        X_text = vectorizer.transform([cleaned_facts])
        print(f"Raw text features shape after vectorization: {X_text.shape}")

        # Step 5: One-hot encode categorical features
        decision_type_one_hot = decision_type_encoder.transform([[features["decision_type"].lower()]])
        disposition_one_hot = disposition_encoder.transform([[features["disposition"].lower()]])

        print(f"Decision type features shape: {decision_type_one_hot.shape}")
        print(f"Disposition features shape: {disposition_one_hot.shape}")

        # Step 6: Combine vectorized text with one-hot encoded features before LDA
        # This is the key change - combine before LDA transformation
        import scipy.sparse
        X_combined_raw = scipy.sparse.hstack([X_text, scipy.sparse.csr_matrix(decision_type_one_hot), 
                                            scipy.sparse.csr_matrix(disposition_one_hot)])
        print(f"Combined raw features shape: {X_combined_raw.shape}")
        
        # Step 7: Apply LDA transformation to the combined features
        if hasattr(lda_model, 'components_') and lda_model.components_.shape[0] > 0:
            # LDA model is already fitted
            X_lda = lda_model.transform(X_combined_raw)
        else:
            # LDA model needs to be fit first (not ideal, but as a fallback)
            X_lda = lda_model.fit_transform(X_combined_raw)
            # Save the fitted model for future use
            joblib.dump(lda_model, "lda_model.pkl")

        print(f"LDA transformed features shape: {X_lda.shape}")
        
        X_combined = X_lda

        print(f"Final features shape: {X_combined.shape}")
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
    pdf_path = "example_legal_case.pdf"  
    result = main(pdf_path)
    print(json.dumps(result, indent=2))