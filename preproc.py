import config
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocessing(df):
    
    df['adults'].fillna(0, inplace = True)
    df['children'].fillna(0, inplace = True)
    df['babies'].fillna(0, inplace = True)
    df['children'] = df['children'].astype(int)

    df['total_guest'] = df['adults'] + df['children'] + df['babies']

    df_hotel = df[['hotel', 'lead_time', 'adults', 'children', 'babies', 'country', 'market_segment', 'distribution_channel', 'is_repeated_guest',
        'previous_cancellations', 'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
        'days_in_waiting_list', 'required_car_parking_spaces', 'total_of_special_requests', 'is_canceled', 'adr', 'total_guest']].copy()

    df_hotel.agent.fillna(0, inplace = True)
    df_hotel.agent = df_hotel.agent.astype(int)

    df_hotel.hotel.replace({"Resort Hotel" : 1, "City Hotel" : 2}, inplace = True)

    # columns with dtype object
    categorical_features = list(df_hotel.select_dtypes(include=['object']).columns)

    # Label Encoder 
    label_encoder_feat = {}
    for i, feature in enumerate(categorical_features):
        df_hotel[feature] = df_hotel[feature].astype(str)
        label_encoder_feat[feature] = LabelEncoder()
        df_hotel[feature] = label_encoder_feat[feature].fit_transform(df_hotel[feature])

    return df_hotel