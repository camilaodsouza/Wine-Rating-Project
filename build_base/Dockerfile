FROM python:3.7.2-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt /opt/build_base/

RUN pip install -r /opt/build_base/requirements.txt

WORKDIR /opt/build_base
