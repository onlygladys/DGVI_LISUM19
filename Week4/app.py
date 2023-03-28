from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
import sklearn
import datetime
from sklearn.preprocessing import StandardScaler 
#import StandardScaler

app = Flask(__name__)
model = pickle.load(open('Usedcarpriceprediction2.pkl', 'rb'))

x =datetime.datetime.now()
@app.route('/')
def home():
    return render_template('index.html')


#standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Age_of_car=int(request.form['Age of car'])

        Transmission=request.form['Transmission']
        if(Transmission=='Manual'):
            Transmission=1
        else:
            Transmission=0
        
        prediction=model.predict([[Year,Present_Price,Kms_Driven2,Age_of_car,Transmission]])
        output=round(prediction[0],2)
       
    return render_template('index.html',prediction_text="The price of the car is {} lakhs".format(output))
    return render_template('index.html')

if __name__=="__main__":
    app.run(port=5000,debug=True)

