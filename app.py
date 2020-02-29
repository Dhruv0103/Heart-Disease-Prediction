import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

global graph
graph = tf.get_default_graph() 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    dj = np.asarray(int_features)
    dj = dj.reshape(1,13)
    with graph.as_default():
        y_hat = model.predict([dj])
        y_hat=float(y_hat[0])

    return render_template('index.html', prediction_text='Heart Disease chances are {} %'.format(y_hat*100))


if __name__ == "__main__":
    app.run(debug=True)
