import joblib
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)


data = pd.read_csv("ext_feature_revised_abs.csv")
dispatch_model = {"RF": "RF.bin", "SVM" : "SVM.bin", "XGB" : "XGB.pkl"}

#Encoding the data
le = LabelEncoder()
labels = data['user']
le.fit(labels)
labels= le.transform(labels)

@app.route('/')
def home():
    return  render_template('home.html')

#for test in postman
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json
    print(data)
    model = request.json["Model"]
    data_ = [j for key, val in data.items() if not key=='Model' for j in val.values()]

    #loading model and scaling
    Xg_model = joblib.load(dispatch_model[model])
    scaling = joblib.load("scaling.bin")

    #make predictions
    new_data = scaling.transform(np.array(data_).reshape(1,-1))
    print(Xg_model.predict(new_data))
    output = le.inverse_transform(Xg_model.predict(new_data))
    return json.dumps(output,default=str)

@app.route('/predict',methods=['POST'])
def predict():
    data = [str(x) for x in request.form.values()]
    model = (data[0]).upper()
    del data[0]

    #loading model and scaling
    Xg_model = joblib.load(dispatch_model[model])
    scaling = joblib.load("scaling.bin")

    #make predictions
    new_data = scaling.transform(np.array(data).reshape(1,-1))
    output = le.inverse_transform(Xg_model.predict(new_data))
    return render_template("home.html", prediction_text= "Model  {}  The person is {} ".format(model, output[0]))

if __name__ == '__main__':
    app.run(debug=True)


