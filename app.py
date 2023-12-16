from joblib import load
import numpy as np
import nltk
import streamlit as st

nltk.download('stopwords')

stopwords = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.porter.PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if str.isalpha(w)]
    tokens = [w for w in tokens if w not in stopwords]
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

def transform_ds(X):
    return np.array(list(map(transform_text, X)))

def length(X):
    return np.array([[len(x)] for x in X])

def densify(X):
    return X.toarray()

model = load(filename="logistic_regressor.joblib")

st.title("SMS spam classifier")

sms = st.text_area("Enter the message : ")

x = np.array([sms])

y = model.predict(x)[0]

if st.button("Predict"):
    if y == 0:
        st.header("Ham")
    else:
        st.header("Spam")