#!/usr/bin/env python
# coding: utf-8

# # Model Building


from platform import python_version
print(python_version())

import py_vollib.black_scholes.implied_volatility as iv


#Import the required Libraries
import numpy as np
import pandas as pd


import pickle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")
from py_vollib.black_scholes.greeks import analytical
import statsmodels.api as sm
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from scipy.stats import norm
from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import options as op

"""
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



raw_data = pd.read_csv("Trady Flow - Best Options Trade Ideas.csv")
pd.options.display.max_columns = None
raw_data.head()



raw_data['Date'] = pd.to_datetime(raw_data['Time']).dt.date


# #### Renaming columns


raw_data.rename(columns={raw_data.columns[1]: 'Symbols', raw_data.columns[2]: 'Option_type', raw_data.columns[3]: 'Expiry', raw_data.columns[4]: 'Strike_price', raw_data.columns[5]: 'Spot_price', raw_data.columns[8]: 'Volume', raw_data.columns[9]: 'Premium'},inplace=True)
raw_data.head()


# #### Handling Datatypes


raw_data['Expiry'] = pd.to_datetime(raw_data['Expiry']).dt.date
raw_data['Date'] = pd.to_datetime(raw_data['Date']).dt.date




repl_dict = {'[kK]': '*1e3', '[mM]': '*1e6', '[bB]': '*1e9', }
raw_data['Volume'] = raw_data['Volume'].replace(repl_dict, regex=True).map(pd.eval)
raw_data['Premium'] = raw_data['Premium'].replace(repl_dict, regex=True).map(pd.eval)



raw_data = raw_data.drop(['Time','OI','ITM'], 1)
raw_data.head()



df_call = raw_data.loc[(raw_data['Option_type'] == 'Call')]



# Replacing 'Call' value with 'c' for Implied Volatility calculation
df_call['Option_type'] = df_call['Option_type'].replace('Call','c')



df_call.isnull().sum()


# #### Calculating 'Time_to_Expiry' col
# Time_to_Expiry = Expiry - Date / 250 (Market functioning days)


df_call['Days_diff'] = (df_call['Expiry'] - df_call['Date']).dt.days
df_call.head()



df_call['Time_to_Expiry'] = (df_call['Days_diff']/250)
df_call.head()



df_call.dtypes



df_call['BidAsk']



df_call.describe()




# Define a function to calculate implied volatility
def calculate_implied_volatility(row):
    S = row['Spot_price']
    K = row['Strike_price']
    r = 0  # Risk-free interest rate (set to 0 for simplicity)
    t = row['Time_to_Expiry']
    price = row['BidAsk']
    option_type = row['Option_type']
    
    try:
        implied_volatility = iv.implied_volatility(price, S, K, t, r, option_type)
    except:
        implied_volatility = None
    
    return implied_volatility

# Apply the function to calculate implied volatility for each row in the DataFrame
df_call['Implied_Volatility'] = df_call.apply(calculate_implied_volatility, axis=1)



df_call['Implied_Volatility'].value_counts


df_call.dtypes




def calculate_delta(row):
    S = row['Spot_price']  # Spot price
    K = row['Strike_price']  # Strike price
    r = 0  # Risk-free interest rate 
    T = row['Time_to_Expiry']  # Time to expiry in years
    sigma = row['Implied_Volatility']  # Implied volatility

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)

    if row['Option_type'] == 'put':
        delta -= 1  # Adjust delta for put options

    return delta

# Apply the delta calculation function to each row in the DataFrame
df_call['Delta'] = df_call.apply(calculate_delta, axis=1)

df_call.head(5)




# Assuming your dataframe is called df_call
for index, row in df_call.iterrows():
    S = row['Spot_price']  # Spot price
    K = row['Strike_price']  # Strike price
    r = 0  # Risk-free interest rate (assumed to be 0 for simplicity)
    t = row['Days_diff'] / 365 # Time to expiry in years
    sigma = row['Implied_Volatility']  # Implied volatility
    option_type = row['Option_type']  # Option type (either 'call' or 'put')

    # Calculate the Delta using the analytical method
    delta_new = analytical.delta('call', S, K, t, r, sigma)
    
    # Assign the calculated Delta value to the dataframe
    df_call.loc[index, 'Delta_new'] = delta_new




# Constants
risk_free_rate = 0  # Example value, replace with your own
df_call['Time_to_Expiry'] = df_call['Time_to_Expiry']

# Calculate Vega
df_call['Vega'] = df_call['Spot_price'] * np.exp(-risk_free_rate * df_call['Time_to_Expiry']) * norm.pdf(df_call['Implied_Volatility']) * np.sqrt(df_call['Time_to_Expiry'])




def calculate_option_price(row):
    S = row['Spot_price']
    K = row['Strike_price']
    T = row['Time_to_Expiry']
    r = 0.05  # Risk-free interest rate
    sigma = row['Implied_Volatility']
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

# Load the dataframe and preprocess the data

# Assuming the dataframe is named 'df'
df_call['Option_call_price'] = df_call.apply(calculate_option_price, axis=1)


df_call = df_call.dropna(axis=0, subset=["Option_call_price"])
df_call["Option_call_price"].isnull().sum()/len(raw_data) * 100



# #### Train-Test split


df_call = df_call.drop(['Symbols','Option_type','Expiry','Volume','Premium','Delta','Delta_new','Vega'], 1)



df_call = df_call.reset_index(drop=True)





# We specify this so that the train and test data set always have the same rows, respectively

df_train, df_test = train_test_split(df_call, train_size = 0.7, test_size = 0.3, random_state = 0)



df_train.drop(['Date'], axis = 1, inplace = True)



df_train.describe()



scaler = MinMaxScaler()



# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['Strike_price', 'Spot_price', 'BidAsk', 'Diff(%)', 'Orders', 'Days_diff', 'Time_to_Expiry', 'Implied_Volatility', 'Option_call_price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



# #### Dividing into X and Y sets for the model building


y_train = df_train.pop('Option_call_price')
X_train = df_train



# ## Building our model
# 
# This time, we will be using the **LinearRegression function from SciKit Learn** for its compatibility with RFE (which is a utility from sklearn)

# ### RFE
# Recursive feature elimination





# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=8)             # running RFE
rfe = rfe.fit(X_train, y_train)



list(zip(X_train.columns,rfe.support_,rfe.ranking_))



col = X_train.columns[rfe.support_]
col



X_train.columns[~rfe.support_]


# ### Building model using statsmodel, for the detailed statistics


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]



# Adding a constant variable 

X_train_rfe = sm.add_constant(X_train_rfe)



lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model



#Let's see the summary of our linear model
print(lm.summary())


# `Orders` is insignificant in presence of other variables; can be dropped


X_train_new = X_train_rfe.drop(["Orders"], axis = 1)



# Adding a constant variable 

X_train_lm = sm.add_constant(X_train_new)



lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model



#Let's see the summary of our linear model
print(lm.summary())



X_train_new.columns



X_train_new = X_train_new.drop(['const'], axis=1)



# Calculate the VIFs for the new model

vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## Residual Analysis of the train data
# 
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.


y_train_price = lm.predict(X_train_lm)




# ## Making Predictions

# #### Applying the scaling on the test sets


num_vars = ['Strike_price', 'Spot_price', 'BidAsk', 'Diff(%)', 'Orders', 'Days_diff', 'Time_to_Expiry', 'Implied_Volatility', 'Option_call_price']

df_test[num_vars] = scaler.transform(df_test[num_vars])


# #### Dividing into X_test and y_test


y_test = df_test.pop('Option_call_price')
X_test = df_test



# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)



# Making predictions
y_pred = lm.predict(X_test_new)


# ## Model Evaluation


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# Evaluate the model to compute the R2
r2_score(y_true=y_test, y_pred=y_pred)


# # Converting to pkl file


pickle.dump(lm, open("model.pkl", "wb"))
lm = pickle.load(open("model.pkl", "rb"))
"""

app = Flask(__name__)

CORS_ALLOW_ORIGIN="*,*"
CORS_EXPOSE_HEADERS="*,*"
CORS_ALLOW_HEADERS="content-type,*"
cors = CORS(app, origins=CORS_ALLOW_ORIGIN.split(","), allow_headers=CORS_ALLOW_HEADERS.split(",") , expose_headers= CORS_EXPOSE_HEADERS.split(","),   supports_credentials = True)

@app.route('/neve/ml/predict', methods=['POST'])
def predict():
    data = request.json
    lm = pickle.load(open("model.pkl", "rb"))

    strike_price = data['strike_price']
    spot_price = data['spot_price']
    bid_ask = data['bid_ask']
    diff = data['diff']
    orders = data['orders']
    days_diff = data['days_diff']
    time_to_expiry = data['time_to_expiry']
    implied_volatility = data['implied_volatility']

    x = np.array([strike_price, spot_price, bid_ask, diff, orders, days_diff, time_to_expiry, implied_volatility])

    y_pred = lm.predict(x)

    return jsonify({'option_call_price': y_pred[0]})

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
    expDate = op.get_expiration_dates(ticker)
    callsData = op.get_calls(ticker,expDate[0]).head(1)
    tickerDetails = {}
    tickerDetails['lastTradeDate'] = callsData['Last Trade Date'].iloc[0]
    tickerDetails['currentPrice'] = callsData['Last Price'].iloc[0]
    tickerDetails['strikePrice'] = callsData['Strike'].iloc[0]
    tickerDetails['expiryDate'] = expDate[0]
    tickerDetails['impliedVolatility'] = callsData['Implied Volatility'].iloc[0]
    tickerDetails['bid'] = callsData['Bid'].iloc[0]
    tickerDetails['ask'] = callsData['Ask'].iloc[0]
    
    return tickerDetails

if __name__ == '__main__':
    #pickle.dump(lm, open("model.pkl", "wb"))
    app.run(host='0.0.0.0',port=5020)

