# Reference: https://www.cnblogs.com/wkang/p/9905444.html

import traceback
import sys

import pandas as pd
from flask import request
from flask import Flask
from flask import jsonify
from sklearn.externals import joblib

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/predict', methods=['POST'])  # Your API endpoint URL would consist /predict
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)
            prediction = list(lr.predict(query))
            return jsonify({'prediction': str(prediction)})
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 8000
    lr = joblib.load('../raw_data/model.pkl')  # Load "model.pkl"
    print('Model loaded')
    model_columns = joblib.load('../raw_data/model_columns.pkl')  # Load "model_columns.pkl"
    print('Model columns loaded')
    app.run(host='127.0.0.1', port=port, debug=True)
