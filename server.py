from collections import deque

from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/recommande',methods=['POST'])
def recommande():
    request_data = request.json
    location = request_data.get('location', '')
    description = request_data.get('description', '')

    recommendations = util.recommender1(location, description)

    return jsonify(recommendations)

if __name__ == "__main__" :
    print("Starting python Flash Server For Home Price Prediction...")
    util.preProcess()
    app.run(debug=True)