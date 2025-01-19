import pandas as pd

def readgreencsv(str):
    df = pd.read_csv(str,skiprows=2,usecols=range(19), header=None)
    df.columns = ['VendorID', 'lpep_pickup_datetime', 'lpep_dropoff_datetime',
       'store_and_fwd_flag', 'RatecodeID', 'PULocationID', 'DOLocationID',
       'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax',
       'tip_amount', 'tolls_amount', 'ehail_fee', 'improvement_surcharge',
       'total_amount', 'payment_type', 'trip_type']
    return df


def clean_data(df):
    #print(df.shape)
    # print(df.head())
    # print(df.columns)
    nacol = columns = ['VendorID', 'lpep_pickup_datetime', 'lpep_dropoff_datetime',
       'store_and_fwd_flag', 'RatecodeID', 'PULocationID', 'DOLocationID',
       'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax',
       'tip_amount', 'tolls_amount', 'improvement_surcharge',
       'total_amount', 'payment_type', 'trip_type']
    df.dropna(axis=0, how='any', subset = nacol ,inplace=True)
    if 'total_amount' in df.columns:
        df = df[df['total_amount'] > 0]
    print(df.head(10))
    return df

path = r"green_tripdata_2016-12.csv"
# green = pd.read_csv(path)
# print(type(green))

data = readgreencsv(path)
data = clean_data(data)
data.to_csv(r"green_tripdata_2016-12.csv",index=False)