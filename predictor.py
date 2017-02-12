#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from transformers import PassengerTransformer


predictor = make_pipeline(
    PassengerTransformer,
    RandomForestClassifier(random_state=123)
)

if __name__ == "__main__":
    import pandas as pd

    train = pd.read_csv('data/cleandata.train.csv')
    test = pd.read_csv('data/cleandata.test.csv')

    predictor.fit(train, train.survived)
    score = predictor.score(test, test.survived)
    print("Score: {:0.4f}".format(score))