from flask import Flask, render_template, request
from flask.templating import _render
import numpy as np
import pickle

from werkzeug.utils import redirect


app = Flask(__name__)
model = pickle.load(open('Heart.pkl', 'rb'))

@app.route('/home', methods=['GET'])
@app.route('/', methods=['GET'])
def Home():
    return render_template('home.html')

@app.route('/graph', methods=['GET'])
def graph():
    return render_template('graph.html')

@app.route('/index',methods=['GET'])
def Index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            values = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
            prediction = model.predict(values)

            return render_template('result.html', prediction=prediction)
    except:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

