FROM continuumio/miniconda

COPY . /app
WORKDIR /app

RUN conda install --yes --file requirements.txt --channel conda-forge

CMD ["python", "run.py"]
