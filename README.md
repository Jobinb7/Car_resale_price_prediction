Project:To predict the resale price of the car  using python.
         
 Data :     car dataset(https://www.kaggle.com/datasets/shalwalsingha/cars-ds-final)
 
            Orginal data:https://drive.google.com/file/d/1ET_9EKbCO7702KIBSFDZur44nOiNQE1A/view?usp=sharing
         
            Cleaned data: https://drive.google.com/file/d/1819qAQzBKXPDDd5P8EnyIa8-nnX0b13N/view?usp=sharing
  Procedure: 

1)   After  preprocessing  the data, it's    imported into the MySQL database "car" as table   "car_ds". Then, by creating a connection from                               Python to  MySQL, the data is fetched into the python platform.

 2)   Cleaning: Initially, the dataset contains 1262 rows and 129 columns, But 48 columns are  dropped  due to the presence of  more than 40 percent of the                 missing values and not relevant to the project. Another 20 columns are also dropped due to the lack of variability.

  3)   Feature engineering: Log transformed the skewed numerical variables like  ex_showroom_price, displacement, height, length and width.

  4)   Preprocessing pipeline:Encoding the categorical variables into numerical format using one-hot encoding and standardize the numerical variable using                    scaling.Here certain string variables like power ,torque  were converted into numerical form.

  5)   Feature selection:For avoid overfitting and to increase the prediction capacity of the model.I have used the Recursive Feature Elimination technique.
     
  6)   Train-Test split:I t’s the process of dividing your dataset into two parts:
       Training set: Used to train the model (typically 70–80% of data).
       Testing set: Used to test the model’s performance (typically 20–30% of data).
  7)    Modelling linear regression: Use pipeline used in   train-test split  with preprocessing steps.
    
  8)    Evaluating the Model: Evaluating the model by various metrics like R square, adjusted R square, mean square error, and mean absolute  error.                           Plotting actual VS prediction scatter plot.

             
 Results: 
 Using Feature selection, more than 40 features were selected. Out of this, the most  dominant features are the  various car models and makers. More  than 55 features  were not selected, which include car models, makers, extra specifications, etc. Here, initially regression models based on raw ex_ showroom_price as the target  variable and regression models  based on log of raw_ex_showroom_price as target variable. Creating two models due to    the very high skewness of ex_showroom_price.   But the log regression model makes predictions with higher error than the raw regression model. 

          

 
 ![](https://github.com/Jobinb7/Car_resale_price_prediction/blob/ddebb743035ae3b598711c1b2265dbfe7a43ef0b/log_raw_regression.PNG?raw=true)        
 ![](https://github.com/Jobinb7/Car_resale_price_prediction/blob/91544c03de0075c460e3de9111ae0644b2212b80/linearRegression_raw1.png)       
 ![](https://github.com/Jobinb7/Car_resale_price_prediction/blob/9f73c0fa99b6ebbc48c6049d5fb5a60771bf9372/Linear_regression_log_2.png)       
         
