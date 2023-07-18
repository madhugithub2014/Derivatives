FROM python:3.9-alpine
COPY ./helloworld.py helloworld.py

ENTRYPOINT ["python", "flask-mcs.py"]
