from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if(int(*prediction) == 0):
        prediction =  "No need to worry, You're Fine."
    else:
        prediction =  "You may have the Disease, Please visit Doctor!"
    
    return render_template('result.html', prediction= prediction)
    
if __name__ == '__main__':
    app.run(debug = True)