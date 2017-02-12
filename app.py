#!/usr/bin/env python
# -*- coding: utf-8 -*-

import types

from flask import Flask
from flask_restful import Api


app = Flask(__name__)
api = Api(app)


def api_route(self, *args, **kwargs):
    def wrapper(cls):
        self.add_resource(cls, *args, **kwargs)
        return cls
    return wrapper

api.route = types.MethodType(api_route, api)
