This is my toy example to test how to serve a prediction pipeline with Flask on a docker container.
I used an example solution to the [Kaggle Titanic challenge](https://www.kaggle.com/c/titanic). It
is a very simple and lightweight implementation. I would eventually improve it, but the purpose of
this miniproject is more focused on the "industralisation" of a machine learning pipeline and not
on finding the best solution to the Kaggle competition.

### How to run it:

- Clone and cd to the repository directory:
```bash
git clone https://github.com/felipeam86/TitanicMicroservice.git
cd TitanicMicroservice
```
- Build an run the docker container:
```bash
docker build -t titanic .
docker run --name titanic_predictor -p 5000:5000 -t titanic
```

And that's it! The pipeline is being served on http://localhost:500/prediction. Here is a Python code to test the REST API:


```python
import requests
import json

def call_prediction_endpoint(passengers, host=None):
    host = host or 'localhost'
    url = "http://{}:5000/prediction".format(host)
    response = requests.post(url=url, json=passengers)#, headers=HEADERS)
    return json.loads(response.json())

passenger_data = [{'passengerid' : "001",
                   'pclass' : "2",
                   'sex' : "male",
                   'age' : "34.0",
                   'sibsp' : "0",
                   'parch' : "0",
                   'fare' : "13.0000"},
                  {'passengerid' : "002",
                   'pclass': "3",
                   'sex': "female",
                   'age': "20.0",
                   'sibsp': "1",
                   'parch': "1",
                   'fare': "15.7417"}]

predictions = call_prediction_endpoint(passenger_data)
print(predictions)
```
