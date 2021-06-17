FROM python:3.7.2-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /opt/clean_data/

RUN pip install -r /opt/clean_data/requirements.txt

WORKDIR /opt/clean_data/