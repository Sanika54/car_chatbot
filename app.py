from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("content/intents.json").read())

words = pickle.load(open('content/words.pkl', 'rb'))
classes = pickle.load(open('content/classes.pkl', 'rb'))
model = load_model('content/chatbot_model.h5')

app = Flask(__name__)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def chatbot_response():
    try:
        msg = request.args.get("msg", "")
        if not msg:
            return jsonify({"reply": "Please send a message"}), 400
            
        intents_list = predict_class(msg)
        if not intents_list:
            return jsonify({"reply": "I'm not sure how to respond to that. Could you please rephrase?"}), 200
            
        result = get_response(intents_list, intents)
        return jsonify({"reply": result})
        
    except Exception as e:
        print(f"Error processing message: {str(e)}")  # For debugging
        return jsonify({"reply": "I apologize, but I'm having trouble processing your request. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
