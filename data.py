import yfinance as yf
import pandas as pd
from fredapi import Fred
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime,  timedelta
from scipy.optimize import minimize
import pickle
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def set_fred_api_key(api_key: str):
    return Fred(api_key=api_key)

def get_latest_price(symbol: str) -> float:
    """
    Get latest historical price for a given symbol using yfinance.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL")
        
    Returns:
        latest historical asset price
    """

    start_date = (datetime.today() - timedelta(days=10)).strftime("%Y-%m-%d")
    end_date = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=end_date)
    closing_price = df["Close"].iloc[-1]

    if df.empty:
        print(f"No data found for {symbol} between {start_date} and {end_date}")
    return closing_price.iloc[0]
def get_price(symbol: str, date: str) -> float:
    """
    Get historical price for a given symbol and date using yfinance.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL")
        date (str): date the stock price is from, (YYYY-MM-DD) format.
        
    Returns:
        historical asset price
    """
  
    start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=10)).strftime("%Y-%m-%d")

    df = yf.download(symbol, start=start_date, end=date)
    closing_price = df["Close"].iloc[-1]

    if df.empty:
        print(f"No data found for {symbol} for {date}")
    return closing_price.iloc[0]

@st.cache_data(ttl=86400)
def get_historical_returns(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get historical log returns data over previous trading days for a given symbol and date period using yfinance.
    
    Args:
        symbol (str): Stock ticker symbol (e.g., "AAPL")
        start_date (str): The date to begin historical returns, (YYYY-MM-DD) form
        end_date (str): The date to end historical returns, (YYYY-MM-DD) form
        
    Returns:
        pd.DataFrame: DataFrame with historical returns
    """
    try:
        print(f"Downloading data for {symbol} at {time.strftime('%X')}")
        df = yf.download(symbol, start=start_date, end=end_date)

        #calculate log returns
        returns = np.log(df['Close'] / df['Close'].shift(1))
        returns = returns.dropna()

        return returns
    
    except Exception as e:
        print(f"Issue downloading data for {symbol} between {start_date} and {end_date}: {e}")
        return returns 
    
    

def fetch_latest_yield(fred, date: str) -> pd.DataFrame:
    """
    Get the yield history between two dates from Fred
    
    Args:
        fred: the fredapi object
        date (str): date we want the rate to be the closest to, (YYYY-MM-DD) format
        
    Returns:
        rates (pd.Dataframe): dataframe of all the dated yield rates from different maturities

    """
    # Mapping: {maturity in years: FRED series ID}
    series_map = {
    0.25: 'DTB3',    # 3-Month
    0.5: 'DTB6',     # 6-Month
    1: 'GS1',        # 1-Year
    2: 'GS2',        # 2-Year
    3: 'GS3',
    5: 'GS5',
    7: 'GS7',
    10: 'GS10',
    20: 'GS20',
    30: 'GS30',
}
    end_date = date
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=31)).strftime("%Y-%m-%d")

    maturities = []
    yields = []
    for maturity, series_id in series_map.items():
        try:
            rate = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            rate = rate.dropna().iloc[-1]
            maturities.append(maturity)
            yields.append(rate / 100) 
            time.sleep(0.4)
        except Exception as e:
            print(f"Skipping {series_id}: {e}")
            time.sleep(0.4)

    return np.array(maturities), np.array(yields)

def nelson_siegel_svensson(t, beta0, beta1, beta2, beta3, tau1, tau2):
    """
    Nelson-Siegel-Svensson model formula
    """
    t = np.asarray(t)

    #some close to zero parameters on yield curves - need padding to avoid runtime errors
    epsilon = 1e-8  # Small value to avoid division by zero
    tau1 = tau1 if abs(tau1) > epsilon else epsilon
    tau2 = tau2 if abs(tau2) > epsilon else epsilon

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            term1 = beta0
            term2 = beta1 * ((1 - np.exp(-t / tau1)) / (t / tau1))
            term3 = beta2 * (((1 - np.exp(-t / tau1)) / (t / tau1)) - np.exp(-t / tau1))
            term4 = beta3 * (((1 - np.exp(-t / tau2)) / (t / tau2)) - np.exp(-t / tau2))
            return term1 + term2 + term3 + term4
        except Warning as w:
            print(f"Caught warning: {w}")
            print("tau1: ", tau1)
            print("tau2 ", tau2)
            print("t: ", t)
            return None

def nss_mse(params, t, y):
    """
    Mean squared error between NSS-predicted and observed yields
    """
    beta0, beta1, beta2, beta3, tau1, tau2 = params
    y_hat = nelson_siegel_svensson(t, beta0, beta1, beta2, beta3, tau1, tau2)
    mse = np.mean((y_hat - y) ** 2)
    return mse

# Fit NSS model
def fit_nss_curve(maturities, yields):
    """
    Fit the NSS model to observed yield data.

    Returns fitted parameters: beta0, beta1, beta2, beta3, tau1, tau2
    """
    initial = [0.02, -0.01, 0.01, -0.01, 1.0, 2.0]
    bounds = [(-1, 1)] * 4 + [(0.01, 10)] * 2
    result = minimize(nss_mse, initial, args=(maturities, yields), bounds=bounds)
    return result.x

def create_yield_curves(fred, dates):
    """
    Create daily yield curves based on available quote days and save them.

    Args:
        fred: the fredapi object
        dates: the pandas dataframe column of distinct dates when options quotes occured

    """
    if fred is None:
        raise RuntimeError("FRED API key not set. Call set_fred_api_key(api_key) first.")
    
    yield_curves = {}
    
    # Get unique dates and sort them
    unique_dates = sorted(dates.unique())
    for date in unique_dates:
        try:
            #get yield data for given date
            maturities, yields = fetch_latest_yield(fred=fred, date=date)
            
            if len(maturities) >= 6:
                #fit the NSS curve to get parameters
                params = fit_nss_curve(maturities, yields)
                yield_curves[date] = params
                print(f"Yield curve created for {date}")
            else:
                print(f"Insufficient yield data for {date}, skipping")
                
        except Exception as e:
            print(f"Error creating yield curve for {date}: {e}")
            continue
    
    return  yield_curves

def get_risk_free_rate_nss(yield_curves, date: str, maturity: float) -> float:
    """
    Get the risk-free rate using NSS method

    Args:
        yield_curves: the database of curve params and dates
        date: The quote day
        maturity (float): the time to maturity you want the risk-free rate to best represent
    """
    
    try:
        
        if date in yield_curves:
            params = yield_curves[date]
            return nelson_siegel_svensson([maturity], *params)[0]
        else:
            print(f"No yield curve found for date {date}")
            return None
    except FileNotFoundError:
        print("No yield curves file found.")
        return None


def get_option_data(ticker, expiration_date: str = None, option_type: str = "call") -> pd.DataFrame:
    """
    Fetch option chain data for a given stock symbol, expiration date, and option type using yfinance.
    
    Args:
        ticker: yfinance object for options
        expiration (str): Expiration date in 'YYYY-MM-DD' format
        option_type (str): 'call' or 'put'
    
    Returns:
        pd.DataFrame: DataFrame containing the option chain for the specified parameters
    """
    try:
    # Get option chain
        options_chain = ticker.option_chain(expiration_date)
        
        if option_type.lower() == 'call':
            return options_chain.calls
        elif option_type.lower() == 'put':
            return options_chain.puts
    except Exception as e:
        print(f"Expiration date {expiration_date} does not exist for option data: {e}")
        return pd.DataFrame() 
        

def preproccessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the option training data to remove problematic rows

    Args:
        data (pd.DataFrame) : option data from the underlying asset
    Returns:
        pd.Dataframe: option data after filtering is complete
    """
    #filter out options that expire earlier than a four days or later than two years
    filtered_maturities = data[(data["T"] > (4 / 365)) & (data["T"] < 2)]

    #filter out options out of 0.8-2.0 moneyness range
    filtered_moneyness = filtered_maturities[(filtered_maturities["M"] > 0.8) & (filtered_maturities["M"] < 2.0)]

    return filtered_moneyness


def get_current_features_values(fred, symbol: str) -> pd.DataFrame:
    """
    Compile all the point in time features of the option contracts from a given symbol
    
    Args:
        symbol (str): Stock ticker symbol (e.g., 'AAPL')
    Returns:
        pd.DataFrame: DataFrame containing the current features and target price for all call option contracts
    """
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if expirations == None:
            raise ValueError(f"There is no option data for symbol: {symbol}")
        
        options = pd.DataFrame()
        #fill up the dataframe with options of all expirations
        for expiration in expirations:
            data = get_option_data(ticker, expiration_date=expiration)

            #calculate time to expiry and current price
            time_diff = (datetime.strptime(expiration, "%Y-%m-%d") - datetime.today()).total_seconds()
            #convert to years until expiry
            data["T"] = time_diff / (24 * 60 * 60 * 365)
            data["V"] = (data["bid"] + data["ask"]) / 2
            

            data = data.rename(columns={"strike": "K"})
            data = data[["K", "T", "V"]]

            
            options = pd.concat([options, data], ignore_index=True)

        S = get_latest_price(symbol=symbol)
        options["S"] = S
        options["M"] = (options["S"] / options["K"])
        options["DATE"] = datetime.today().strftime("%Y-%m-%d")
        #get the yield curves for these dates
        yield_curves = create_yield_curves(fred, options["DATE"])
        options["r"] = options.apply(
                    lambda row: get_risk_free_rate_nss(yield_curves=yield_curves, date=row["DATE"], maturity=(row["T"] / 365)),
                    axis=1
                )
        #reorder
        options = options[["DATE", "S", "T", "K", "V", "r", "M"]]

        filtered_options = preproccessing(options)
        return filtered_options
    except Exception as e:
        print(f"Error fetching option data: {e}")
        return pd.DataFrame()

def data_fetch(data_dir: str, delimiter=","):
    """
    Fetch all data from text files in the given directory and its subfolders into a single DataFrame.
    """
    # exclude greeks and other unecessary columns
    cols_to_read = [
        " [QUOTE_DATE]", " [UNDERLYING_LAST]", " [DTE]", " [C_SIZE]",
        " [C_BID]", " [C_ASK]", " [STRIKE]"
    ]
    col_types = {
        " [QUOTE_DATE]": str,
        " [UNDERLYING_LAST]": float,
        " [DTE]": float,
        " [C_SIZE]": str,
        " [C_BID]": float,
        " [C_ASK]": float,
        " [STRIKE]": float,
    }
    #renamed cols
    rename_map = {
        " [QUOTE_DATE]": "DATE",
        " [UNDERLYING_LAST]": "S",
        " [DTE]": "T",
        " [C_SIZE]": "SIZE",
        " [C_BID]": "BID",
        " [C_ASK]": "ASK",
        " [STRIKE]": "K",
    }

    all_options = []
    #load the yield curves in
    with open("yield_curves.pkl", "rb") as file:
        yield_curves = pickle.load(file)
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                options = pd.read_csv(file_path, delimiter=delimiter, usecols=cols_to_read, dtype=col_types, na_values=['', ' '])
                options = options.rename(columns=rename_map)
                #filter out put options and unrealistic call options
                size_split = options["SIZE"].str.split("x", expand=True)
                left = size_split[0].str.strip().astype(int)
                right = size_split[1].str.strip().astype(int)
                options = options[(left != 0) & (right != 0)]
                options = options[options["BID"] != 0.0]
                options = options[options["T"] > 0]

                #use T in terms of years
                options["T"] = options["T"] / 365
                #remove the extra space in date values
                options["DATE"] = options["DATE"].str.strip()
                #calculate option value from mean bid/ask price and drop those cols
                options["V"] = (options["BID"] + options["ASK"]) / 2
                #calculate moneyness - ratio of underlying to strike
                options["M"] = (options["S"] / options["K"])
                #apply the risk free rate function to each row in order to create the r column
                options["r"] = options.apply(
                    lambda row: get_risk_free_rate_nss(yield_curves=yield_curves, date=row["DATE"], maturity=row["T"]),
                    axis=1
                )
                finished_df = options.drop(["BID", "ASK", "SIZE"], axis=1) 
                #add all dataframes to the accumulating data structure
                all_options.append(finished_df)
                print(f"Finished fetch for {file_path}.")
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
    if all_options:
        raw_data = pd.concat(all_options, ignore_index=True)
        processed_data = preproccessing(raw_data)
        return processed_data
    else:
        print("No text files found.")
        return pd.DataFrame()

def plot_variable_distributions(df, columns=None, output_dir="variable_distributions"):
    """
    Plot and save the distribution of each variable in the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (list, optional): List of columns to plot. If None, plot all numeric columns.
        output_dir (str): Directory to save the plots.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    os.makedirs(output_dir, exist_ok=True)
    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.show()
        plt.close()

    


