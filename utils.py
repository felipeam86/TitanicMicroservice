
import requests
import json

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
