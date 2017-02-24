#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import requests
import json

from predictor import predictor
from explainer import construct_predictor_explainer

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 5000

def call_explain_endpoint(df,
                          host=DEFAULT_HOST,
                          port=DEFAULT_PORT,
                          html=False):

    endpoint = "explainhtml" if html else "explain"
    response = requests.post(
        url=get_url(host, port, endpoint),
        json=df_to_json(df)
    )
    if html:
        return response.content.decode()
    else:
        return json.loads(response.json())


def call_prediction_endpoint(df,
                             host=DEFAULT_HOST,
                             port=DEFAULT_PORT):

    response = requests.post(
        url=get_url(host, port, "prediction"),
        json=df_to_json(df)
    )
    return json.loads(response.json())


def get_url(host, port, endpoint):
    return "http://{}:{}/{}".format(host, port, endpoint)


def df_to_json(df):
    cols = ["passengerid", "pclass", "sex", "age", "sibsp", "parch", "fare"]
    return df[cols].astype(str).to_dict(orient='records')


def serialized_prediction(df):

    response = pd.DataFrame(np.vstack([predictor.predict(df),
                                       predictor.predict_proba(df).max(axis=1)]).T,
                            columns=["Survived", "Probability"],
                            index=df['passengerid'])

    return response.to_json(orient='index')


predictor_explainer = construct_predictor_explainer(show_notebook=False)


def explain_prediction(df):
    df = df.convert_objects(convert_numeric=True)
    exp = predictor_explainer.transform(df)
    response = pd.DataFrame(dict(exp.as_list()),
                            index=df['passengerid'])
    return response.to_json(orient="index")


def explain_prediction_html(df):
    df = df.convert_objects(convert_numeric=True)
    exp = predictor_explainer.transform(df)
    return exp.as_html()
