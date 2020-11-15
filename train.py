import pandas as pd
import config
import preproc
import cross_val
import model_dispatcher
from sklearn import metrics
import joblib
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

def run(folds, model):
        
    # Loading the dataset
    df = pd.read_csv(config.TRAINING_FILE)

    # Preprocessing
    df = preproc.preprocessing(df)

    # cross validation (Stratified K folds)
    df = cross_val.create_folds(df)

    # training and validation set
    df_train = df[df.kfold != folds].reset_index(drop = True)
    df_valid = df[df.kfold == folds].reset_index(drop = True)

    X_train = df_train.drop(columns = ['is_canceled']).values
    y_train = df_train.is_canceled.values

    X_valid = df_valid.drop(columns = ['is_canceled']).values
    y_valid = df_valid.is_canceled.values

    clf = model_dispatcher.models[model]

    if model != 'dnn':
        print ("Training...")
        clf.fit(X_train, y_train)

        print ("Done!!")
        preds = clf.predict(X_valid)
    else:
        clf.summary()
        print ("Training...")

        clf.fit(
            X_train, y_train,
#           validation_data = (X_valid, y_valid),
            batch_size = config.batch_size,
            epochs = config.epochs
#           callbacks = [model_dispatcher.early_stopping],
            )

        print ("Done!!")
        preds = clf.predict(X_valid)
        preds = (preds>0.5).astype(int)
 
    acc = metrics.accuracy_score(y_valid, preds)
    print ("Fold = {} Accuracy = {}".format(folds, acc))
    print ("-------Classification Report")
    print (metrics.classification_report(y_valid, preds))

    if model !='dnn':
        joblib.dump(
            clf,
            os.path.join(config.MODEL_OUTPUT, f"{model}_fold_{folds}.bin")
        )
    else:
        name = "dnn_model_fold_" + str(folds) + str(".h5")
        clf.save(config.MODEL_OUTPUT + name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        type = int
    )

    parser.add_argument(
        "--model",
        type = str
        
    )

    args = parser.parse_args()

    run(
        folds = args.folds,
        model = args.model
    )

