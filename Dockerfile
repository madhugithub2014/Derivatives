FROM python:3.10.2-buster

# Installing packages
# RUN apk update
# RUN apk add py-pip
# RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install scipy
RUN pip install flask
RUN pip install -U flask-cors
RUN pip install pandas
RUN pip install yahoo_fin
# RUN pip install matplotlib
# RUN pip install yahoo_fin

WORKDIR /home/dpmluser2/neve
COPY . /home/dpmluser2/neve

ENTRYPOINT ["python3", "predict_option_call_price.py"]
