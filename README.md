# EDA-SA-and-Hotel-Booking-Cancellation-Prediction

The objective of the project is to predict the hotel booking status of the guest if it'll be canceled or not based on the various features like ADR (Average Daily Rate), booking changes, lead time, type of the hotel booked, and more. The type of hotels given in the dataset are Resort Hotels and City Hotels.

##### config.py - Configuration File</br>
##### cross_val.py - Cross Validation File to make Stratifiedkfolds (5 folds)</br>
##### preproc.py - Preprocessing the Data</br>
##### model_dispatcher.py - ML and DL Models</br>
##### train.py - Main Run file</br>

### DATASET
#### The dataset used in the project is taken from kaggle - https://www.kaggle.com/jessemostipak/hotel-booking-demand </br>
#### The article on the dataset - https://www.sciencedirect.com/science/article/pii/S2352340918315191#f0005 </br>

### EDA and SA
The Exploratory Data Analysis and Statistical Analysis is done to get the insight about the data and answer some questions for example "Which is the busiest month of the year ?", "What is the average price of the room per person per night ?" and more. The correlation heatmap is plotted as well to see the most important features and the threshold correlation taken is 0.04 and the features are taken on the basis of that.

The EDA and SA notebook is given in the notebooks folder of the repository. [use the jupyter nbviewer to see the interactive graphs.]

### MODELLING

Four Models are trained for the purpose of the prediction named Logistic Regression, Random Forest, XGBoost and Deep Neural Network. The models are trained and validated on a 5 fold Cross validation set [Stratified k folds]. You can see the hyperparameters and the structure of the models in the model dispatcher file.
Some trained models are saved in the models folder.

### How to run
1. Download the repository.
2. open the terminal and cd to the repository.
3. type:
    python train.py --folds 0 --model logistic_regression

Note - you can change the folds from 0 to 4.
       The models name can be seen in the models dictionary in the model_dispatcher, but I can give them here.
       - logist
