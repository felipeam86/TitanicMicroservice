# -*- coding: utf-8 -*-

"""
REST API serving a simple Titanic survival predictor
"""

import pandas as pd

from flask import request
from flask_restful import Resource

from app import app, api
from model import PassengerSchema
from predictor import serialized_prediction

__author__ = "Felipe Aguirre Martinez"
__email__ = "felipeam86@gmail.com"

passenger_schema = PassengerSchema(many=True, strict=True)


@api.route('/prediction')
class Prediction(Resource):
    def post(self):
        json_data = request.get_json()
        result = passenger_schema.load(json_data)
        df = pd.concat(result.data)
        return serialized_prediction(df)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', processes=4)