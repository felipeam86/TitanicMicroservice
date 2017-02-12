FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y --force-yes python python-dev python-setuptools software-properties-common gcc python-pip
RUN apt-get clean all

# Flask Port
EXPOSE 5000

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "run.py"]
