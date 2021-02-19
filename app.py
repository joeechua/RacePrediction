from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle as p
import json

modelfile = 'race_prediction.pickle'
model = p.load(open(modelfile, 'rb'))

app = Flask(__name__)

bangsa = ['MELAYU', 'CINA', 'INDIA', 'LAIN-LAIN']

@app.route('/race', methods = ['GET'])
def get_races():
    return {'races': bangsa}

@app.route("/", methods = ['GET'])
def hello():
    return render_template('index.html')

@app.route("/race/<name>", methods = ['GET'])
def get(name):
    index = model.predict([name])[0]
    p_race = bangsa[index]
    return {'predictedrace': p_race}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
