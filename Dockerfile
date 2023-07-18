FROM python:3.9-alpine
COPY ./flask-mcs.py flask-mcs.py

ENTRYPOINT ["python", "flask-mcs.py"]
