import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# Model building for chatbot (training.py file)

import numpy as np
import tensorflow as tf
import random
import json

# Chatbot application (chatbot.py file)

import nltk
from nltk.stem import WordNetLemmatizer
import pickle

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model("chatbotmodel.h5")
bot_name = "Dex"

# cleaning sentences
def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
  return sentence_words

# bag of words
def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  # initializing bag full of zeros for all individual words
  bag = [0] * len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

# predicting result
def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.25
  results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

  results.sort(key=lambda x : x[1], reverse = True)
  return_list = []
  for r in results:
    return_list.append({'intent':classes[r[0]],'probability': str(r[1])})
  return return_list

def get_response(intents_list, intents_json):
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break
    else:
      result = "I do not understand..."
  return result

print("Let's chat! type 'quit' to exit")

while True:
  message = input("You: ")
  if message == "quit":
      break

  ints = predict_class(message)
  res = get_response(ints, intents)
  print(f"{bot_name}: " + res)