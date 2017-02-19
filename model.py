#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Passenger schema for the REST API
"""

import pandas as pd
from marshmallow import Schema, fields, post_load


class PassengerSchema(Schema):
    passengerid = fields.Str(required=True)
    pclass = fields.Integer(required=True)
    sex = fields.Str(required=True)
    age = fields.Str(required=True)
    sibsp = fields.Str(required=True)
    parch = fields.Str(required=True)
    fare = fields.Str(required=True)

    @post_load
    def make_offer(self, data):
        return pd.DataFrame(data, index=[0])


if __name__ == '__main__':
    passenger_schema = PassengerSchema(many=True, strict=True)
    passenger_data = [{'passengerid': "001",
                       'pclass': "2",
                       'sex': "male",
                       'age': "34.0",
                       'sibsp': "0",
                       'parch': "0",
                       'fare': "13.0000"},
                      {'passengerid': "002",
                       'pclass': "3",
                       'sex': "female",
                       'age': "20.0",
                       'sibsp': "1",
                       'parch': "1",
                       'fare': "15.7417"}]

    result = passenger_schema.load(passenger_data)
    df_deserialized = pd.concat(result.data)
    print(df_deserialized.to_string())
