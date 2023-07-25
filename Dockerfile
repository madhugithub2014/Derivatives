FROM python:3.10.2-buster

# Installing packages
# RUN apk update
# RUN apk add py-pip
# RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install scipy
RUN pip install flask
RUN pip install -U flask-cors
RUN pip install 'pandas<2.0.0'
RUN pip install yahoo_fin
RUN pip install matplotlib
RUN pip install py_vollib
RUN pip install seaborn
RUN pip install statsmodels
RUN pip install -U scikit-learn

WORKDIR /home/dpmluser2/neve
COPY . /home/dpmluser2/neve

ENTRYPOINT ["python3", "predict_option_call_price.py"]
