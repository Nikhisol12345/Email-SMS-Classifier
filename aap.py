import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  Y = []
  for i in text:
    if i.isalnum():
      Y.append(i)

  text = Y[:]
  Y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      Y.append(i)

  text = Y[:]
  Y.clear()
  for i in text:
    Y.append(ps.stem(i))

  return " ".join(Y)
tfidf = pickle.load(open('vectorizer (1).pkl','rb'))
model = pickle.load(open('model (1).pkl','rb'))
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")
if st.button('Predict'):
  # preprocess
  transform_sms = transform_text(input_sms)

  # vectorize
  vector_input = tfidf.transform([transform_sms])
  # predict
  result = model.predict(vector_input)
  # Display
  if result == 1:
      st.header("Spam")
  else:
      st.header("Not Spam")
