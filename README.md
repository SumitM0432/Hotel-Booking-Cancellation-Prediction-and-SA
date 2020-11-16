# EDA-SA-and-Hotel-Booking-Cancellation-Prediction

The objective of the project is to predict the hotel booking status of the guest if it'll be cancelled or not based on the various features like ADR (Average Daily Rate), booking changes, lead time, type of the hotel booked, and more. The type of hotels given in the dataset is Resort Hotels and City Hotels.

#### Kaggle Notebook : [https://www.kaggle.com/sumitm004/eda-sa-and-hotel-booking-cancellation-prediction]

##### config.py - Configuration File</br>
##### cross_val.py - Cross Validation File to make Stratifiedkfolds (5 folds)</br>
##### preproc.py - Preprocessing the Data</br>
##### model_dispatcher.py - ML and DL Models</br>
##### train.py - Main Run file</br>
##### predict.py - Run Trained model for prediction</br>

### DATASET
#### The dataset used in the project is taken from kaggle - https://www.kaggle.com/jessemostipak/hotel-booking-demand </br>
#### The article on the dataset - https://www.sciencedirect.com/science/article/pii/S2352340918315191#f0005 </br>

### EDA and SA
The Exploratory Data Analysis and Statistical Analysis is done to get the insight about the data and answer some questions for example "Which is the busiest month of the year ?", "What is the average price of the room per person per night ?" and more. The correlation heatmap is plotted as well to see the most important features and the threshold correlation is taken is 0.04 and the features are taken based on that.</br>

The EDA and SA notebook is given in the notebooks folder of the repository. [use the jupyter nbviewer to see the interactive graphs.]

### MODELLING

Four Models are trained for the prediction named Logistic Regression, Random Forest, XGBoost and Deep Neural Network. The models are trained and validated on a 5 fold Cross-validation set [Stratified k folds]. You can see the hyperparameters and the structure of the models in the model dispatcher file.
Some trained models are saved in the models folder. you can train your model or use the trained model to predict using the predict.py.

### How to run
[TRAIN]</br>
1. Download the repository.
2. open the terminal and cd to the repository.
3. type:</br>
    python train.py --folds 0 --model logistic_regression

[PREDICTION]
1. Download the repository.
2. open the terminal and cd to the repository.
3. Check the name of the trained models in the models folder.
3. type:</br>
    python predict.py --folds 0 --saved_model xgboost_fold_0.bin

###### Note:
you can change the folds from 0 to 4.</br>
For training the models name can be seen in the models dictionary in the model_dispatcher.py, but I can give them here.</br>
- logistic_regression</br>
- xgboost</br>
- random_forest</br>
- dnn</br>

For predicting check the saved model name in the models folder and use the name as shown above.
