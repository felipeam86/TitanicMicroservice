#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.pipeline import TransformerMixin, BaseEstimator, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

from transformers import PassengerTransformer


class Explainer(TransformerMixin, BaseEstimator):
    """
    Pipeline friendly Lime explainer
    """

    def __init__(self, feature_names, class_names, predict_proba, notebook=True):
        self.feature_names = feature_names
        self.class_names = class_names
        self.predict_proba = predict_proba
        self.notebook = notebook

    def fit(self, X, y=None):
        self.explainer_ = LimeTabularExplainer(
            X,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=True
        )
        return self

    def transform(self, X):
        assert X.shape[0] == 1, "Can only explain one prediction at a time"
        X = X.flatten()
        exp = self.explainer_.explain_instance(X, self.predict_proba)
        if self.notebook:
            exp.show_in_notebook(show_table=True, show_all=False)
        return exp


train = pd.read_csv('data/cleandata.train.csv')
test = pd.read_csv('data/cleandata.test.csv')

rf = RandomForestClassifier(random_state=123)
rf.fit(PassengerTransformer.fit_transform(train), train.survived)


def construct_predictor_explainer(show_notebook=False):
    predictor_explainer = make_pipeline(
        PassengerTransformer,
        Explainer(
            feature_names=['age', 'sex', 'pclass', 'sibsp', 'parch', 'fare'],
            class_names=list(train.survived.unique()),
            predict_proba=rf.predict_proba,
            notebook=show_notebook
        )
    )

    return predictor_explainer.fit(train)


if __name__ == "__main__":
    predictor_explainer = construct_predictor_explainer(show_notebook=True)
    exp = predictor_explainer.transform(test.sample(1))
