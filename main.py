from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # load the model
    model = joblib.load("my_model.pkl")

    # get the data from the request
    data = request.get_json()

    # make predictions with the model
    predictions = model.predict(data)

    # return the predictions as a JSON object
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
