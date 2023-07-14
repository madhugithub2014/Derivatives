FROM python:3.9-alpine
COPY ./Options-premium-pred.ipynb Options-premium-pred.ipynb

ENTRYPOINT ["python", "Options-premium-pred.ipynb"]
