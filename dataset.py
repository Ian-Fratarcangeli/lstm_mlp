import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.utils import resample
from data import get_historical_returns
from datetime import datetime

class OptionDataset(Dataset):
    def __init__(self, option_df, log_returns_df, seq_length):
        """
        option_df: pd.DataFrame with ['DATE', 'T', 'S', 'r', 'K', 'M', 'V']
        log_returns_df: pd.DataFrame with SPY historical log returns indexed by date
        seq_length: number of log returns to extract before option date
        """
        self.option_df = option_df.reset_index(drop=True)
        self.log_returns_df = log_returns_df
        self.seq_length = seq_length


    def __len__(self):
        return len(self.option_df)

    def __getitem__(self, idx):
        row = self.option_df.iloc[idx]
        end_date = row["DATE"]
        try:
            end_date = pd.to_datetime(end_date)
            # Find the closest previous trading day
            available_dates = self.log_returns_df.index
            closest_date = available_dates[available_dates <= end_date].max()
            if pd.isna(closest_date):
                raise ValueError(f"No trading days before {end_date}")
            end_idx = self.log_returns_df.index.get_loc(closest_date)
        except KeyError:
            raise ValueError(f"Error finding log returns for {end_date}.")
        start_idx = end_idx - self.seq_length
        log_returns_seq = self.log_returns_df.iloc[start_idx:end_idx].values
        log_returns_seq = log_returns_seq.reshape(-1, 1).astype(np.float32)
        # MLP input
        mlp_features = np.array([row["T"], row["S"], row["r"], row["K"], row["M"]], dtype=np.float32).reshape(1, -1)
        mlp_features = mlp_features.flatten()
        # Target
        target = np.array([row["V"]], dtype=np.float32).reshape(1, -1)
        target = target.flatten()
        return torch.tensor(log_returns_seq), torch.tensor(mlp_features), torch.tensor(target)

def data_process(bins=6, alpha=0.4, beta=0.7, random=42 ):
    options_df = pd.read_pickle("all_options_data.pkl")

    returns_df = get_historical_returns(symbol="SPY", start_date="2018-01-01", end_date=datetime.today().strftime("%Y-%m-%d"))

    #filter to get more recent options, don't need that much training data
    #take out contracts valued under $1, harder for model to fit
    start = "2021-6-01"
    end = "2023-12-31"
    filtered_options_df = options_df[
        (pd.to_datetime(options_df["DATE"]) >= pd.to_datetime(start)) &
        (pd.to_datetime(options_df["DATE"]) <= pd.to_datetime(end)) &
        (options_df["V"] >= 1.0)
    ].copy()
    n_bins = bins
    filtered_options_df['price_bin'] = pd.cut(filtered_options_df['V'], bins=n_bins, labels=False, duplicates='drop')
    # Find the size of the largest bin
    max_bin_size = filtered_options_df['price_bin'].value_counts().max()

    # Upsample each bin, downsample the biggest bin
    sampled_frames = []
    for bin_id in range(n_bins):
        bin_df = filtered_options_df[filtered_options_df['price_bin'] == bin_id]
        num = len(bin_df)

        if num == max_bin_size:
            downsample_target = int(num * beta)
            downsampled_bin = resample(bin_df, 
                                     replace=False,
                                     n_samples=downsample_target,
                                     random_state=random)

            sampled_frames.append(downsampled_bin)
        else:
            #upsample proportionally to largest bin and size of current bin
            upsample_factor = (max_bin_size / num) ** alpha
            n_samples = int(num * upsample_factor)

            upsampled_bin = resample(
                bin_df,
                replace=True,
                n_samples=n_samples,
                random_state=42
            )

            sampled_frames.append(upsampled_bin)

    #combine the upsampled bins
    upsampled_train_df = pd.concat(sampled_frames).drop(columns=['price_bin'])

    # Shuffle the upsampled DataFrame
    upsampled_train_df = upsampled_train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # transform features with minmax
    feature_cols = ["T", "S", "r", "K", "M"]
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    transformed_features = feature_scaler.fit_transform(upsampled_train_df[feature_cols])
    upsampled_train_df.loc[:, feature_cols] = transformed_features

    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    V_transformed = pt.fit_transform(upsampled_train_df[["V"]])
    upsampled_train_df.loc[:, ["V"]] = V_transformed

    transformed_target = target_scaler.fit_transform(upsampled_train_df[["V"]])
    upsampled_train_df.loc[:, ["V"]] = transformed_target

    length = len(upsampled_train_df)
 
    train_size = round(0.85 * length)
    val_size = round(0.97 * length)
  
    train_df = upsampled_train_df[0:train_size]
    val_df = upsampled_train_df[train_size:val_size]
    test_df = upsampled_train_df[val_size:length]

    return train_df, val_df, test_df, returns_df, target_scaler, pt, feature_scaler