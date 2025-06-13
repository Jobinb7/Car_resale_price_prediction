Project:To predict the resale price of the car  using python.
         
 Data :  Data set : car dataset(https://www.kaggle.com/datasets/shalwalsingha/cars-ds-final)
 
         Orginal data:https://drive.google.com/file/d/1ET_9EKbCO7702KIBSFDZur44nOiNQE1A/view?usp=sharing
         
         Cleaned data: https://drive.google.com/file/d/1819qAQzBKXPDDd5P8EnyIa8-nnX0b13N/view?usp=sharing
         
  Procedure: 1) After  preprocessing  the data, it's    imported into the MySQL database "car" as table "car_ds".
              Then, by creating a connection from Python to MySQL, the data is fetched into the python platform.
              
             2) Cleaning: Initially, the dataset contains 1262 rows and 129 columns, But 48 columns are  dropped  due to the presence of  more than
              40 percent of the missing values and not relevant to the project. Another 20 columns are also dropped due to the lack of variability.

            3) Feature engineering: Log transformed the skewed numericl variables like  ex_showroom_price, displacement, height, length and width.

             4)Preprocessing pipeline:Encoding the categorical variables into numerical format using one-hot encoding and standardize the nunerical
               variable using scaling.Here certain string variables like power ,torque  were converted into numerical form.
             
             5)  Feature selection:For avoid overfitting and to increase the prediction capacity of the model.I have used the Recursive Feature Elimination
                 techique.
              6) Train -Test split :It’s the process of dividing your dataset into two parts:
              
                                    Training set: Used to train the model (typically 70–80% of data)

                                    Testing set: Used to test the model’s performance (typically 20–30% of data)
             
               
