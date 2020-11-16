import joblib
import config
import cross_val
import pandas as pd
import argparse
from sklearn import metrics
from tensorflow import keras



def predict(folds, model):

    df = pd.read_csv(config.VALIDATION_FILE)

    #validation set
    df_valid = df[df.kfold == folds].reset_index(drop = True)

    X_valid = df_valid.drop(columns = ['is_canceled']).values
    y_valid = df_valid.is_canceled.values

    if 'dnn' not in str(model):
        mod = joblib.load(config.MODEL_OUTPUT + str(model))
        
        predictions = mod.predict(X_valid)
    else:
        mod = keras.models.load_model(config.MODEL_OUTPUT + str(model))
        
        predictions = mod.predict(X_valid)
        predictions = (predictions>0.5).astype(int)

    acc = metrics.accuracy_score(y_valid, predictions)
    print ("Fold = {} Accuracy = {}".format(folds, acc))
    print ("-------Classification Report")
    print (metrics.classification_report(y_valid, predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        type = int
    )

    parser.add_argument(
        "--saved_model",
        type = str
    )

    args = parser.parse_args()

    predict(
        folds = args.folds,
        model = args.saved_model
    )
