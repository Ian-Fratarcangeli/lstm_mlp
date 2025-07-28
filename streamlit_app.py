import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from model import OptionPricingModel
from dataset import data_process
from data import get_price, set_fred_api_key
from datetime import datetime
from dotenv import load_dotenv

st.set_page_config(page_title="LSTM-MLP Option Pricer", layout="wide")
st.title("LSTM-MLP Option Pricer Dashboard")

load_dotenv()
API_KEY = os.getenv("FRED_API_KEY")
fred = set_fred_api_key(api_key=API_KEY)

# --- Introduction ---
st.header("Introduction")
st.write("""The purpose of this project is to test the LSTM-MLP framework against option data to gain a pattern recognition advantage over traditional methods of option pricing. 
         Traditional methods, like Black-Scholes, use an implied volatility to estimate the option value, which we can find with the market's price of the option.
          The Heston model instead implements volatility through a stochastic process of its own that is then fed into the asset price process, 
         requiring some parameters like long-term volatility. Both of these methods can be seen in the Monte Carlo Simulation project, 
         and when you visit the dashboard you'll notice that the parameters I've mentioned are significant in each of their option pricing evaluations. 
         It is very difficult to accurately estimate parameters that represent the asset price movements; 
         stochastic processes are most common but market changes can render successful models to useless. 
         The LSTM-MLP model is meant to surpass traditional methods by learning important asset price metrics rather than parameterizing the distributions. 
         In doing so, my hope is that the LSTM will sift through the returns data and collect important information relevant to an option's future value 
         (possibly information applicable to the implied volatility). The MLP will then couple the LSTM information with important characteristics of the contract to 
         provide an accurate pricing of the option. 
""")

#data loading in
train_df, val_df, test_df, returns_df, target_scaler, pt, feature_min_max = data_process(bins=20, alpha=0.2, beta=0.7)
st.header("Data")
st.write("""
         Call option contracts on SPY are collected from OptionsDX as daily quotes between the dates of January 1st, 2021 and December 31st, 2023. The returns for each contract 
         were fetched from Yahoo Finance and converted to logarithmic returns. Data for risk-free rates were collected by generating yield curves from yields of that time using the 
         Nelson-Siegel-Svensson model. Options with a time to maturity of less than 4 days or greater than two years, or with moneyness below 0.8 and above 2.0
         were removed from the dataset to minimize outlier bias. All of the option features were transformed using a min-max scaler to avoid 
         inappropriate model weights based on different value distributions.
         The option value (V) distribution is highly skewed towards low option price contracts, creating a target imbalance resulting in difficulty predicting 
         higher option prices. To combat this, the data is upsampled for higher option price contracts to balance price bins, and the target is given a 
         Yeo-Johnson transform to administer more normality in the distribution. Overall, approximately 1,750,000 contracts were gathered to be split for training, validation, and testing.
""")

with st.expander("Show Sample Data"):
    st.write("Sample of Training Data:")
    st.dataframe(train_df.head(10))
    st.write("Sample of SPY Returns Data:")
    st.dataframe(returns_df.head(10))


with st.expander("Variable Distributions"):
    feature_cols = ["S", "K", "M", "V"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    for i, col in enumerate(feature_cols):
        ax = axes[i // 2, i % 2]
        ax.hist(train_df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    
    plt.tight_layout()
    st.pyplot(fig)
# --- Model Architecture ---
st.header("Model Architecture")
st.write("""
The model consists of two main components:
- **LSTM**: Processes a sequence of historical log returns of the underlying asset for each option.
- **MLP**: Takes in the LSTM output in addition to classic option features (T = time to maturity, S = underlying price, r = risk-free rate, K = strike price, M - moneyness).
The output of the MLP is the predicted option price (V).
""")
with st.expander("Show Model Code"):
    with open("model.py", "r") as f:
        st.code(f.read(), language="python")
st.write("""The LSTM portion of the model consists of three layers and the MLP has four layers, in which batch normalization and dropout are implemented.
         A Leaky-Relu activation function is used within each layer to maintain negative option prices close to 0 while still encouraging learning in model training.
         A Softplus activation function is chosen over Leaky-Relu for the final layer to ensure the outputted option prices are not negative.
        """)
# --- Training Process ---
st.header("Training Process")
st.write("""The model is trained using Huber loss and AdamW optimizer for more momentum. Early stopping is enabled to avoid overfitting and a learning rate scheduler is implemented to better control gradient flow later in the training process.
        The first 85% of the data is used to train, while 12% is for validation and the final 3% for testing. 
         The data is sliced sequentially for the training/validation/testing split so the model can learn patterns over the given period of time and apply them to the validation/testing sets in the later periods.  
""")
#put training images instead of code here
with st.expander("Show training images"):
    col1, col2 = st.columns(2)
    with col1:
        st.image("training_images/learning_curve.png", caption="Training and Validation loss over Epochs", use_container_width=True)
    with col2:
        st.image("training_images/gradients.png", caption="Gradients for model weights over Epochs", use_container_width=True)

st.write("""Although the model exhibits learning in the training set,
         evaluation on the validation set did not produce a curve that clearly resembles improvement over epochs. 
         It can be argued that there was slight improvement in the validation loss up to five epochs and then some overfitting that may have occured later.
         For this reason, stopping was triggered early at five epochs of training and those weights were saved and loaded into this dashboard. Excluding a couple small spikes
         over the training, the gradient flow stayed controlled and did not exhibit any vanishing or exploding.
          """)

# --- Results & Evaluation ---
st.header("Results & Evaluation")
# Load model and evaluate on test set
def load_model(model_path, device):
    model = OptionPricingModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Load model
model_path = "option_pricing_model_final.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, device)


#Lets do some writing here
st.write(f"The testing phase was ran with {len(test_df)} option contracts. A mean absolute percent error (MAPE) of 9.02% was calculated for the evaluation.")
st.write("""Even with the upsampling, the model still had issues with consistently underestimating higher priced option contracts. This problem is visualized in the
         percent error vs. V chart that illustrates a slight bend downwards in percent error as option price increases. Model error stayed relatively consistent
         over maturity, minus some sparseness for contracts with quick maturities (less than a week). For contracts out of and at the money (M<=1), the error shows a lot
         of variablity because these contract prices are very cheap, therefore small misses in estimates result in large percent errors. The in the money (M>1)
         contracts exhibit the downward bend similar to the first visual, which I'm assuming also had to do with the lack of data for call options far in the money, and therefore
         high in price.   
""")
st.subheader("Model Evaluation Visuals")
col1, col2, col3 = st.columns(3)
with col1:
    st.image("inference_plots/price_error.png", caption="Model percent error over Option Price (V)", use_container_width=True)
with col2:
    st.image("inference_plots/perror_maturity.png", caption="Model percent error over Maturity (T)", use_container_width=True)
with col3:
    st.image("inference_plots/perror_moneyness.png", caption="Model percent over Moneyness (M)", use_container_width=True)

st.header("Discussion")
st.write("""Overall, I'm pleased with the performance of the LSTM-MLP model on option prices. Because the data I gathered for call options over time was free, the quality
         of this data may have contributed to many of the issues present in training and evaluation. I chose not to go over two million option contracts because the training was time-consuming and resulted in 
         computation strain for my computer. Nonetheless, the model produced a strong MAPE and was able
         to learn from financial data that is volatile and skewed. Below I have loaded in the trained model for an interactive dashboard where you can make inputs and ask for an 
         estimated call option value. It is important to keep in mind that the data I used to train the model stopped at the end of 2023; the model has not learned market changes since then that 
         might impact today's market prices, but hopefully it can extrapolate.
""")

# --- Interactive Prediction ---
st.header("Interactive Option Price Prediction")
st.write("Input option parameters to get a model price prediction for a SPY call option:")
error = 0
with st.form("predict_form"):
    Date = st.text_input("Quote Date of Contract (YYYY-MM-DD)", value = datetime.today().strftime("%Y-%m-%d"))
    # Validate input
    valid_date = None
    if Date:
        try:
            valid_date = datetime.strptime(Date, "%Y-%m-%d").date()

            if valid_date > datetime.today().date():
                st.error("Quote dates cannot be in the future. Please try again.")
                error +=1
            else: st.success(f"Valid date: {valid_date}")
        except ValueError:
            st.error("Invalid date format. Please enter date as YYYY-MM-DD.")
            error +=1
    else:
        st.error("Need to enter a valid quote date.")
        error +=1

    T = st.number_input("Time to Maturity (T, in days)", min_value=0.01, max_value=3.0, value=0.5, step=0.01)
    r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.1, value=0.04, step=0.001)
    K = st.number_input("Strike Price (K)", min_value=1.0, max_value=1000.0, value=400.0, step=1.0)
    submitted = st.form_submit_button("Predict Option Price")

if submitted & (error != 1):
    try:
        # Scale features
        S = get_price(symbol="SPY", date=Date)
        M = S / K
        features = np.array([[T, S, r, K, M]], dtype=np.float32)
        features_scaled = feature_min_max.transform(features)
        try:
            date = pd.to_datetime(Date)
            # Find the closest previous trading day
            available_dates = returns_df.index
            closest_date = available_dates[available_dates <= date].max()
            if pd.isna(closest_date):
                raise ValueError(f"No trading days before {Date}")
            end_idx = returns_df.index.get_loc(closest_date)
        except KeyError:
            raise ValueError(f"Error finding log returns for {Date}.")
        start_idx = end_idx - 140
        log_returns_seq = returns_df.iloc[start_idx:end_idx].values
        log_returns_seq = log_returns_seq.reshape(-1, 1).astype(np.float32)
   
        log_returns_tensor = torch.tensor(log_returns_seq).unsqueeze(0).to(device)
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            output = model(log_returns_tensor, features_tensor)
            pred_original_scale = target_scaler.inverse_transform(
                pd.DataFrame(output.cpu().numpy(), columns=['V'])
            )
            pred_original = pt.inverse_transform(
                pd.DataFrame(pred_original_scale, columns=['V'])
            )
            st.success(f"Predicted Call Option Price: ${pred_original.flatten()[0]:.4f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

st.caption("""Project uses PyTorch, Pandas, and scikit-learn.""")