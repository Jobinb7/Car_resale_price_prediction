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
 Using Feature selection, more than 40 features were selected. Out of this, the most  dominant features are the  various car models and makers. More  than 55 features  were not selected, which include car models, makers, extra specifications, etc. 

          

 
 ![](https://github.com/Jobinb7/Car_resale_price_prediction/blob/2710229eb017e19ca53344502ef9b95d7141e689/correct_prediction.PNG)
 ![](https://github.com/Jobinb7/Car_resale_price_prediction/blob/91544c03de0075c460e3de9111ae0644b2212b80/linearRegression_raw1.png)       
        
  Here training R ^2 is higher than the testing R^2 ,But most errors are under 30,000 Rupees .This shows the model generlaize well.
  Because of this high training R^2 shows small overfitting only.On average,  model underestimates or overestimates the resale price
  by ₹7.19 lakhs. RMSE = 17.98 lakhs is much higher than MAE, which suggests that there are a few large errors (outliers) that are 
  penalized more by RMSE.
 ![](https://github.com/Jobinb7/Car_resale_price_prediction/blob/df285f65fe439322b94d03a77f38407574ae9e38/Top25_features.png)
 ![](https://github.com/Jobinb7/Car_resale_price_prediction/blob/ad616c7a60e87d4a92f16bb8929d5cc2fcd57d0c/coefficient_selected_featues.PNG)
 
 Top Positive Influences on Resale Price:
                                  
      
Feature	Coefficient (₹)	Interpretation
model_Mulsanne	₹1.18 Cr	The Bentley Mulsanne model contributes significantly to resale value due to its ultra-luxury status.
make_Bentley	₹1.12 Cr	Bentley as a brand has a very high brand value, boosting resale price across its models.
make_Ferrari	₹81.7 Lakh	Ferrari’s brand value leads to strong resale retention.
model_Range	₹80.0 Lakh	Generic Range Rover models generally hold high resale prices.
log_length	₹42.7 Lakh	Longer vehicles (SUVs/premium sedans) tend to have higher resale value; this log-transformed feature captures that trend.

Top Negative Influences on Resale Price:


model_"Range Velar  (-₹72.7 L)             This specific Range Rover model lowers resale value.
make_Skoda	  (-₹64.6 L)	       Skoda company  has less demand.
make_Ford	  (-₹59.9 L)	       Ford  company has less demand.
model_Xuv500	   (-₹56.0 L)	       Mahindra model has  low resale price.
model_Mustang	   (-₹52.7 L)	       Surprisingly, Mustang model of Ford  also has low demand.

Summary:
Luxury brands (Bentley, Ferrari, Aston Martin) and their flagship models greatly increase resale value.

Mass-market brands (Skoda, Ford, Tata) or models that are discontinued or outdated lower the predicted price.

Technical specs and features, when transformed (log), have a reasonable effect but are less influential than brand/model.
