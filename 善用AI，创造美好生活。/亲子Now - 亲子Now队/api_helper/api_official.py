# Reference: https://www.cnblogs.com/wkang/p/9905444.html

import traceback
import sys

import pandas as pd
from flask import request
from flask import Flask
from flask import jsonify
from sklearn.externals import joblib
from recommender.baseline_rs import main as recommender_system

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/predict', methods=['POST'])  # Your API endpoint URL would consist /predict
def predict():
    try:
        json_ = request.json


        temp = pd.DataFrame(json_)

        model_columns = ['topic', 'url']
        query = temp.reindex(columns=model_columns, fill_value=0)
        result = []
        for index, row in query.iterrows():
            result.append(recommender_system(row['topic']))

        return jsonify({'prediction': str(result)})
    except:
        return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 8000

    app.run(host='127.0.0.1', port=port, debug=True)
