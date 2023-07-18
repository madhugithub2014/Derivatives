FROM python:3.9-alpine

# Installing packages
RUN apk update
RUN apk add py-pip
RUN apk add --no-cache python3-dev 
RUN pip install --upgrade pip
RUN pip install numpy scipy pandas matplotlib yahoo_fin

COPY ./flask-mcs.py flask-mcs.py

ENTRYPOINT ["python3", "flask-mcs.py"]
