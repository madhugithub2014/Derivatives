FROM python:3.9-alpine

# Installing packages
RUN apk update
RUN apk add py-pip
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install scipy
RUN apk add --update --no-cache py3-numpy py3-pandas@edge   
RUN pip install matplotlib
RUN pip install yahoo_fin

COPY ./flask-mcs.py flask-mcs.py

ENTRYPOINT ["python3", "flask-mcs.py"]
