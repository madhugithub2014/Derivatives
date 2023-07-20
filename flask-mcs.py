#!/usr/bin/env python
# coding: utf-8

# In[1]:
#from IPython import get_ipython

#get_ipython().system('pip3 install flask')

# In[2]:


from flask import Flask, jsonify, request
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from yahoo_fin import options as op


# In[3]:


app = Flask(__name__)


# In[4]:


@app.route("/")
def simulator_service():
    return "Up & Running !"


@app.route("/mcs/<string:option_type>")
def derivative_option(option_type):
    result = {
        "simulator type":"Monte Carlo Simulation",
        "Option type":option_type
    }
    return jsonify(result)

@app.route("/mcs/price/<spotPrice>/<strikePrice>/<time>/<volatility>/<steps>/<trials>")
def calculate_mcs(spotPrice,strikePrice,time,volatility,steps,trials):
       
        spotPrice = float(spotPrice)
        strikePrice = float(strikePrice)
        T = int(time)
        volatility = float(volatility)
        steps = int(steps)
        trials = int(trials)
        
        r = 0
        q = 0;
        
        paths= geo_paths(spotPrice,T,r,q,volatility,steps,trials)
        payoffs = np.maximum(paths[-1]-strikePrice, 0)
        option_price = np.mean(payoffs)*np.exp(-r*T)
        
        return jsonify(option_price) # return data with 200 OK


# In[5]:


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
    ST = np.log(S) +  np.cumsum(((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt) * np.random.normal(size=(steps,N))),axis=0)
    
    return np.exp(ST)


# In[6]:


@app.route("/symbols")
def get_symbols():
    sp_wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp_wiki_df_list = pd.read_html(sp_wiki_url)
    sp_df = sp_wiki_df_list[0]
    return jsonify(list(sp_df['Symbol'].values))


# In[7]:


ticker = 'AAPL'
expDate = op.get_expiration_dates(ticker)
callsData = op.get_calls(ticker,expDate[0]).head(1)
print("for ticker => ",callsData)


# In[8]:


callsData.columns


# In[9]:


@app.route("/symbol/details/<ticker>")
def get_ticker_details(ticker):
    expDate = op.get_expiration_dates(ticker)
    callsData = op.get_calls(ticker,expDate[0]).head(1)
    return callsData.to_json()


# In[ ]:


if __name__ == '__main__':
    #app.run()  # run our Flask app
    app.run(host='0.0.0.0',port=5000)


# In[ ]:




