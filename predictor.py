#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from transformers import PassengerTransformer


train = pd.read_csv('data/cleandata.train.csv')
test = pd.read_csv('data/cleandata.test.csv')


predictor = make_pipeline(
    PassengerTransformer,
    RandomForestClassifier(random_state=123)
)
predictor.fit(train, train.survived)


def serialized_prediction(df):

    response = pd.DataFrame(np.vstack([predictor.predict(df),
                                       predictor.predict_proba(df).max(axis=1)]).T,
                            columns=["Survived", "Probability"],
                            index=df['passengerid'])

    return response.to_json(orient='index')

if __name__ == "__main__":

    score = predictor.score(test, test.survived)
    print("Score: {:0.4f}".format(score))

    random_prediction = serialized_prediction(test.sample(1))
    print("Prediction: {0!r}".format(random_prediction))