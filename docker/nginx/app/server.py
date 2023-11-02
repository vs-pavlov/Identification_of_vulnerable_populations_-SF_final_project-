from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# загружаем модель из файла
with open('/app/models/pipeline.pkl', 'rb') as pkl_file:
    model = pickle.load(pkl_file)


# создаём приложение
app = Flask(__name__)

@app.route('/')
def index():
    msg = "Тестовое сообщение. Сервер запущен!"
    return msg

@app.route('/predict', methods=['POST'])
def predict():
    features = request.json
    # using loc methods
    with open('/app/models/cluster_df.pkl', 'rb') as pkl_file:
        cluster_df = pickle.load(pkl_file)
    cluster_df.iloc[-1] = features
    cluster_df['pred_cluster'] = model.predict(cluster_df)
    return jsonify({
        'prediction': float(cluster_df['pred_cluster'].iloc[-1])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)