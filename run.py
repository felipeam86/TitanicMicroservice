# -*- coding: utf-8 -*-

"""
REST API serving a simple Titanic survival predictor
"""

import argparse

import pandas as pd
from flask import request
from flask_restful import Resource

from app import app, api
from model import PassengerSchema
from utils import serialized_prediction, explain_prediction, explain_prediction_html

__author__ = "Felipe Aguirre Martinez"
__email__ = "felipeam86@gmail.com"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = "5000"

passenger_schema = PassengerSchema(many=True, strict=True)


@api.route('/prediction')
class Prediction(Resource):
    def post(self):
        json_data = request.get_json()
        result = passenger_schema.load(json_data)
        df = pd.concat(result.data)
        return serialized_prediction(df)


@api.route('/explain')
class Explain(Resource):
    def post(self):
        json_data = request.get_json()
        result = passenger_schema.load(json_data)
        df = pd.concat(result.data)
        return explain_prediction(df)


@app.route('/explainhtml', methods=['POST'])
def explain_html():
    json_data = request.get_json()
    result = passenger_schema.load(json_data)
    df = pd.concat(result.data)
    return explain_prediction_html(df)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-H", "--host", default=DEFAULT_HOST,
                        help="Hostname for the REST API [default {}]".format(DEFAULT_HOST))

    parser.add_argument("-P", "--port", default=DEFAULT_PORT,
                        help="Port for the REST API [default {}]".format(DEFAULT_PORT))

    parser.add_argument("-n", "--cpus", type=int, default=4)

    parser.add_argument("-d", "--debug", action="store_true", dest="debug")

    parser.add_argument("-p", "--profile", action="store_true", dest="profile")

    args = parser.parse_args()

    if args.profile:
        from werkzeug.contrib.profiler import ProfilerMiddleware

        app.config['PROFILE'] = True
        app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
        args.debug = True

    app.run(
        debug=args.debug,
        host=args.host,
        port=int(args.port),
        processes=int(args.cpus)
    )
