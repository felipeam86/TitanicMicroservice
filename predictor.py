#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from transformers import PassengerTransformer


train = pd.read_csv('data/cleandata.train.csv')
test = pd.read_csv('data/cleandata.test.csv')


predictor = make_pipeline(
    PassengerTransformer(),
    RandomForestClassifier(random_state=123)
)
predictor.fit(train, train.survived)

if __name__ == "__main__":

    score = predictor.score(test, test.survived)
    print("Predictor score: {:0.4f}\n".format(score))

    sample = test.sample(1)
    print("Passenger id: {}".format(sample.passengerid.iloc[0]))
    print("Survived: {}".format(predictor.predict(sample)[0]))
    print("Probability: {}".format(predictor.predict_proba(sample).max(axis=1)[0]))
