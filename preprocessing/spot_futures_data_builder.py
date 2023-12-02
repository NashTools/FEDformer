import datetime

import numpy as np
import pandas as pd
import json

from preprocessing.feature_generation import FeatureFactory


# Function to add 1 microsecond to duplicates
def resolve_duplicates(index):
    counter = 0
    new_index = []
    latest = index[0] - pd.Timedelta(microseconds=1)
    for dt in index:
        counter += 1
        if dt <= latest:
            dt = latest + pd.Timedelta(microseconds=1)
        latest = dt
        new_index.append(dt)
        # print counter every 10000
        if counter % 10000 == 0:
            print(counter)
    return new_index


def read_spot(path, start=None, end=None, res_dup=True):
    nrows = end - start if end is not None else None
    df = pd.read_csv(path, skiprows=start, nrows=nrows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")
    if res_dup:
        df.index = resolve_duplicates(df.index)
    return df


def read_futures(path, start=None, end=None, res_dup=True):
    nrows = end - start if end is not None else None
    df = pd.read_csv(path, skiprows=start, nrows=nrows)
    df["event_time"] = pd.to_datetime(df["event_time"], unit="ms")
    df = df.set_index("event_time")
    df = df.sort_index()
    df = df.sort_values(by=["update_id"])
    if res_dup:
        df.index = resolve_duplicates(df.index)
    return df


def standardize_column_names(df, col_aliases):
    new_cols = []
    for col in df.columns:
        found = False
        for key, aliases in col_aliases.items():
            if col in aliases:
                new_cols.append(key)
                found = True
                break
        if not found:
            new_cols.append(col)
    df.columns = new_cols
    return df


def create_spot_dataset():
    data = read_spot("../dataset/BTCUSDT-bookTicker-2023-06-s")
    print("read data done")
    with open("col_aliases.json", "r") as f:
        mapping = json.load(f)

    factory = FeatureFactory(mapping)
    factory.apply_features(data, ["spread", "log_return"])
    standardize_column_names(data, mapping)
    # select only the columns we need
    data = data[["bid_price", "bid_volume", "ask_price", "ask_volume", "mid_price", "spread", "log_return"]]
    data.to_pickle("../dataset/Spot-2023-06.pkl")


def create_futures_dataset():
    data = read_futures("../dataset/BTCUSDT-bookTicker-2023-06.csv")
    print("read data done")
    with open("col_aliases.json", "r") as f:
        mapping = json.load(f)

    factory = FeatureFactory(mapping)
    factory.apply_features(data, ["spread", "mid_price"])
    standardize_column_names(data, mapping)
    # select only the columns we need
    data = data[["bid_price", "bid_volume", "ask_price", "ask_volume", "mid_price", "spread"]]
    data.to_pickle("../dataset/Futures-2023-06.pkl")


def combine_spot_futures_datasets(s_df, f_df):
    s_df.columns = [col + "_s" for col in s_df.columns]
    f_df.columns = [col + "_f" for col in f_df.columns]

    combined_df = pd.concat([s_df, f_df], axis=1)
    print("concat done")

    combined_df["log_return_s"] = combined_df["log_return_s"].fillna(0)
    combined_df.fillna(method="ffill", inplace=True)
    combined_df["efp"] = combined_df["mid_price_f"] - combined_df["mid_price_s"]
    combined_df["efp"] = combined_df["efp"].round(4)
    combined_df.dropna(inplace=True)
    return combined_df


def main1():
    column_names = pd.read_csv("../dataset/Spot-2023-06.csv", nrows=0).columns

    # Create a dictionary with all column data types set to numpy.float16
    dtype_dict = {col: np.float32 for col in column_names}
    dtype_dict[column_names[0]] = str

    # Now, read the CSV again with the specified data types
    df = pd.read_csv("../dataset/Spot-2023-06.csv", dtype=dtype_dict, index_col=0)
    df.index = pd.to_datetime(df.index)

    # change datatype from second column to one before last column to float16
    df.to_pickle("../dataset/Spot-2023-06.pkl")


def merge_spot_futures_data_main():
    spot = pd.read_pickle("../dataset/Spot-2023-06.pkl")
    spot.dropna(inplace=True)
    spot = spot.loc[datetime.datetime(2023, 6, 25):]
    print("read spot done")
    futures = pd.read_pickle("../dataset/Futures-2023-06.pkl")
    futures.dropna(inplace=True)
    futures = futures.loc[datetime.datetime(2023, 6, 25):]
    print("read futures done")
    futures = combine_spot_futures_datasets(spot, futures)
    print("combine done")
    futures.to_pickle("../dataset/Futures-Spot-2023-06-after-25.pkl")


# main
if __name__ == "__main__":
    d = pd.read_pickle("../dataset/Futures-Spot-2023-06-before-15.pkl")
    cols = list(d.columns.values)
    cols.pop(cols.index("log_return_s"))
    d = d[cols + ["log_return_s"]]
    d = d.loc[:, ~d.columns.str.contains("price")]
    # select rows where date is before 8th of June
    train = d.loc[:"2023-06-08"]
    train.to_pickle("../dataset/seven-days-train.pkl")
    val = d.loc["2023-06-08":"2023-06-09"]
    val.to_pickle("../dataset/seven-days-val.pkl")
    test = d.loc["2023-06-09":"2023-06-10"]
    test.to_pickle("../dataset/seven-days-test.pkl")