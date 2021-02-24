from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import pickle as p
import json

modelfile = 'race_prediction.pickle'
model = p.load(open(modelfile, 'rb'))

app = Flask(__name__)

bangsa = ['MELAYU', 'CINA', 'INDIA', 'LAIN-LAIN']

@app.route('/api/race', methods = ['GET'])
def get_races():
    return {'races': bangsa}

@app.route("/")
def hello():
    return render_template('index2.html')

@app.route("/api/race/<name>", methods = ['GET'])
def get_race(name):
    if request.method =='GET':
        index = model.predict([name])[0]
        p_race = bangsa[index]
        return {'predictedrace': p_race}
    else:
        return {'message': 'incorrect request method'}
    
@app.route("/api", methods = ['GET'])
def explain_api():
    return render_template('index.html')
    
@app.route("/race/predict", methods = ['GET'])
def pred_race():
   if request.method == 'GET':
       name = (request.args['Pname']).lower()
       race_index = (model.predict([name]))[0]
       probs = model.predict_proba([name])[0]
       maxi = np.where(probs == np.max(probs))[0][0]
       return render_template('result.html', prediction=race_index, m = probs[0], c = probs[1], i = probs[2] ,l = probs[3], maxim = maxi)
   else:
       return {'message': 'incorrect request method'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
