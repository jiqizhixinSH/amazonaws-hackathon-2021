FROM python:3.8

MAINTAINER shengxio@ualberta.ca
USER root

WORKDIR /app

ADD __init__.py
ADD engine.py
ADD FileControl.py
ADD nlp_engine.pkl
ADD README.md
ADD requirements.txt
ADD UI.py
ADD UI-control.js
ADD UI-style.js
ADD utilities.py
ADD Dockerfile
ADD city_SanJose_Minutes.csv /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 8051
ENV NAME stickers

CMD["screamlit","run","UI.py"]