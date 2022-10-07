from urllib import response
import nltk
import json
import pickle
import numpy as np
import random
import tensorflow
from data_preprocessing import bag_of_words_encoding, get_stem_words
ignore_words = ['?', '!',',','.', "'s", "'m"]
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

intents = json.loads(open('./intents.json').read())
words=pickle.load(open('./words.pkl','rb'))
classes=pickle.load(open('./classes.pkl','rb'))
def preprocess_user_input(user_input):
    input_word_token_1=nltk.word_tokenize(user_input)
    input_word_token_2=get_stem_words(input_word_token_1,ignore_words)
    input_word_token_2=sorted(list(set(input_word_token_2)))
    bag=[]
    bag_of_words=[]
    for word in words:
        if word in input_word_token_2:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
        
    bag.append(bag_of_words)
    return np.array(bag)

def bot_class_pridiction(user_input):
    inp=preprocess_user_input(user_input)
    pridiction=model.predict(inp)
    predicted_class_label=np.argmax(pridiction[0])
    return predicted_class_label
def bot_response(user_input):
    predicted_class_label=bot_class_pridiction(user_input)
    predicted_class=classes[predicted_class_label]
    for intent in intents['intents']:
        if intent['tag']==predicted_class:
            bot_response=random.choice(intent['responses'])
            return bot_response

print('hello i am stella how can i help you')
while True:
    user_input=input("type your message hear")
    print("user input",user_input)
    response=bot_response(user_input)
    print("bot response",response)

