#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from platform import python_version
print(python_version())

import py_vollib.black_scholes.implied_volatility as iv
import socket

#Import the required Libraries
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import options as op
from yahoo_fin import stock_info as info
print(pd.__version__)
app = Flask(__name__)

CORS_ALLOW_ORIGIN = "*"
CORS_EXPOSE_HEADERS = "*,*"
CORS_ALLOW_HEADERS = "content-type,*"

# Initialize CORS with the desired configuration for the '/neve/*' endpoint
#cors = CORS(app, resources={r"/neve/*": {"origins": ["http://localhost:9000", "http://127.0.0.1:5015"]}})
CORS(app, resources={r"/neve/*": {"origins": "*"}})

lm = pickle.load(open("model.pkl", "rb"))

# Endpoint for predicting option call price
@app.route('/neve/ml/predict', methods=['POST'])
def predict():
    data = request.json
    strike_price = data['strike_price']
    spot_price = data['spot_price']
    bid_ask = data['bid_ask']
    diff = data['diff']
    orders = data['orders']
    days_diff = data['days_diff']
    time_to_expiry = data['time_to_expiry']
    implied_volatility = data['implied_volatility']
    

    x = np.array([strike_price, spot_price, bid_ask, diff, orders, days_diff, time_to_expiry, implied_volatility])


    # Reshape x to a 2-dimensional array with shape (1, 8)
    x_reshaped = x.reshape(1, -1)

    #Ensure x_reshaped has the appropriate data type
    x_reshaped = x_reshaped.astype(float)  # Change 'float' to the appropriate data type

# Now you can use x_reshaped to make predictions with the model
    y_pred = lm.predict(x_reshaped)
    print(y_pred[0])
    return jsonify({'option_call_price': y_pred[0]})

# Endpoint for Monte Carlo simulation
@app.route("/")
def simulator_service():
    return "Up & Running !"

@app.route("/neve/mcs/<string:option_type>")
def derivative_option(option_type):
    result = {
        "simulator type":"Monte Carlo Simulation",
        "Option type":option_type
    }
    return jsonify(result)


@app.route("/neve/callOption/price",methods=['POST'])
def calculate_mcs():
       
        data = request.json
        strikePrice = float(data['strikePrice'])
        spotPrice = float(data['spotPrice'])
        T = float(data['time'])
        volatility = float(data['volatility'])
        steps = int(data['steps'])
        trials = int(data['trials'])
       
        r = 0.05
        q = 0;
       
        paths= geo_paths(spotPrice,T,r,q,volatility,steps,trials)
        payoffs = np.maximum(paths[-1]-strikePrice, 0)
        option_price = np.mean(payoffs)*np.exp(-r*T)
       
        return jsonify(round(option_price, 2))

    
def geo_paths(S, T, r, q, sigma, steps, N):
    """
    Inputs
    #S = Current stock Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = risk free interest rate
    #q = dividend yield
    # sigma = volatility 
    
    Output
    # [steps,N] Matrix of asset paths 
    """
    dt = T/steps
    ST = np.log(S) +  np.cumsum(((r - q - sigma**2/2)*dt +\
                              sigma*np.sqrt(dt) * \
                              np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)

@app.route("/neve/symbols")
def get_symbols():
    sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp_wiki_df_list = pd.read_html(sp_wiki_url)
    sp_df = sp_wiki_df_list[0]
    return jsonify(list(sp_df['Symbol'].values))

@app.route("/neve/symbol/details/<ticker>")
def get_ticker_details(ticker):
    openPrice = info.get_live_price(ticker)
    expDate = op.get_expiration_dates(ticker)
    callsData = op.get_calls(ticker,expDate[5])
    tickerDetails = {}
    for i in range(len(callsData)):
        if float(callsData.loc[i, "Implied Volatility"].replace('%','')) > 0 :
            tickerDetails['lastTradeDate'] = callsData.loc[i, "Last Trade Date"]
            tickerDetails['currentPrice'] = openPrice
            tickerDetails['strikePrice'] = callsData.loc[i, "Strike"]
            tickerDetails['expiryDate'] = expDate[5]
            tickerDetails['impliedVolatility'] = callsData.loc[i, "Implied Volatility"]
            tickerDetails['bid'] = callsData.loc[i, "Bid"]
            tickerDetails['ask'] = callsData.loc[i, "Ask"]
        break
    return tickerDetails

# Get the hostname
#hostname = socket.gethostname()

# Print the hostname (optional)
#print("Hostname:", hostname)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5015)
   #app.run(debug=True, use_reloader=False, port=5096)
   #app.run()


# In[ ]:





# In[ ]:




