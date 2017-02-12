#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


if __name__ == "__main__":

    score = predictor.score(test, test.survived)
    print("Score: {:0.4f}".format(score))

