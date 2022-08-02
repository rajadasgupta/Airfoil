import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
from flask import Response
from flask_cors import CORS
import pandas as pd
import numpy as np


app=Flask(__name__)
pickled_model = pickle.load(open("model_rf.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
    #Reading the data from the JSON. In this case passed from Postman
        data = request.json['data']
        print(data)
    #a. the model accepts the the data in 2D; b. the JSON is passed as dictionary and the values are captured and passed to model
        new_data = [list(data.values())]
        output = pickled_model.predict(new_data)[0]
        return jsonify(output)
    except Exception as e:
        raise Exception ("The error recieved is :", e)

@app.route('/predict', methods=['POST'])
def predict():
    try:
    #Reading the data from the web page. In this case passed from website
        web_data = [float(x) for x in request.form.values()]
        
    #a. the model accepts the the data in 2D; b. the data is passed is converted to 2D list and passed to model
        final_data = [np.array(web_data)]
        print(final_data)
        output_web = pickled_model.predict(final_data)[0]
        return render_template('home.html',prediction_text=f"Airfoil pressure level is {output_web}")

    except Exception as e:
        raise Exception ("The error recieved is :", e)

if __name__=="__main__":
    app.run(debug=True)
