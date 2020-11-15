import config
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from tensorflow.keras import layers

# DNN model
def dnn():
    dnn = keras.Sequential([
        layers.BatchNormalization(input_shape = config.input_shape),
        layers.Dense(512, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation = 'relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(256, activation = 'relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation = 'sigmoid')
    ])

    dnn.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy']
    )
    return dnn

# early_stopping = keras.callbacks.EarlyStopping(
#         patience = config.patience,
#         min_delta = config.min_delta,
#         restore_best_weights = True,
#     )

models = {
    'dnn' : dnn(),
    'logistic_regression' : LogisticRegression(C = 1.2),
    'xgboost': XGBClassifier(eta = 0.35, max_depth = 12),
    'random_forest' :  RandomForestClassifier(n_estimators = 120)
    }