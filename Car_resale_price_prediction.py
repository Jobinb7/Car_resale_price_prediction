# ---------------------------------
# IMPORTS
# ---------------------------------
import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------

def extract_city_mileage(value):
    try:
        if pd.isnull(value):
            return np.nan
        match = re.search(r'([\d\.]+)', value)
        if match:
            return float(match.group(1))
    except:
        return np.nan
    return np.nan

def extract_extended_warranty(value):
    try:
        if pd.isnull(value):
            return np.nan
        match = re.search(r'(\d+)', value)
        if match:
            return float(match.group(1))
    except:
        return np.nan
    return np.nan

def extract_powerps(value):
    try:
        if pd.isnull(value):
            return np.nan
        match = re.search(r'(\d+)PS', value)
        if match:
            return float(match.group(1))
    except:
        return np.nan
    return np.nan

# ---------------------------------
# LOAD DATA FROM MYSQL
# ---------------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Jobin1995",
    database="car"
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM car_ds")
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=[col[0] for col in cursor.description])
cursor.close()
conn.close()

# ---------------------------------
# FEATURE ENGINEERING
# ---------------------------------
df['city_mileage_numeric'] = df['city_mileage'].apply(extract_city_mileage)
df['extended_warranty_numeric'] = df['extended_warranty'].apply(extract_extended_warranty)
df['power_numeric'] = df['power'].apply(extract_powerps)

log_transform_cols = ['ex_showroom_price', 'displacement', 'height', 'length', 'width']

for col in log_transform_cols:
    if col in df.columns:
        df[f'log_{col}'] = np.log1p(df[col])
    else:
        print(f"Warning: Column {col} not found!")

categorical_cols = [
    'make','model',
    'emission_norm','ventilation_system','abs','ebd','average_fuel_consumption',
    'central_locking','child_safety_locks','engine_malfunction_light','front_brakes','fuel_type','engine_immobilizer'
]

y_raw = df['ex_showroom_price']


X_num_log = df[[f'log_{col}' for col in log_transform_cols if f'log_{col}' in df.columns]]
X_num_raw = df[['city_mileage_numeric', 'extended_warranty_numeric','power_numeric', 'seating_capacity']]
X_num = pd.concat([X_num_log, X_num_raw], axis=1)
X_cat = df[categorical_cols]

df_model_raw = pd.concat([X_num, X_cat, y_raw], axis=1).dropna()
X_num_clean_raw = df_model_raw[X_num.columns.tolist()]
X_cat_clean_raw = df_model_raw[categorical_cols]
y_clean_raw = df_model_raw['ex_showroom_price']



# ---------------------------------
# DEFINE PREPROCESSING PIPELINE
# ---------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X_num.columns.tolist()),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), X_cat.columns.tolist())
    ]
)

# ---------------------------------
# FEATURE SELECTION (RFE)
# ---------------------------------
X_all_raw = preprocessor.fit_transform(pd.concat([X_num_clean_raw, X_cat_clean_raw], axis=1))
base_model = LinearRegression()
rfe = RFE(estimator=base_model, n_features_to_select=40)
rfe.fit(X_all_raw, y_clean_raw)

# 1️Get feature names
numeric_features = X_num.columns.tolist()

# Important: fit preprocessor so OneHotEncoder has the categories
preprocessor.fit(pd.concat([X_num_clean_raw, X_cat_clean_raw], axis=1))

# Get one-hot encoded feature names
onehot_encoder = preprocessor.named_transformers_['cat']
onehot_feature_names = onehot_encoder.get_feature_names_out(X_cat.columns.tolist())

# Full feature names (order matches X_all_raw columns)
full_feature_names = numeric_features + list(onehot_feature_names)

# Selected feature names
selected_feature_names = np.array(full_feature_names)[rfe.support_]
non_selected_feature_names = np.array(full_feature_names)[~rfe.support_]
# Print them
print("\n===== Selected Features (RFE) =====")
for feature in selected_feature_names:
    print(f"- {feature}")

print("\n===== Non-Selected Features (RFE) =====")
for feature in non_selected_feature_names:
    print(f"- {feature}")




# ---------------------------------
# TRAIN RAW TARGET MODEL
# ---------------------------------
model_pipeline_raw = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    pd.concat([X_num_clean_raw, X_cat_clean_raw], axis=1),
    y_clean_raw,
    test_size=0.2,
    random_state=42
)
#
model_pipeline_raw.fit(X_train_raw, y_train_raw)
y_train_pred_raw = model_pipeline_raw.predict(X_train_raw)

y_pred_raw = model_pipeline_raw.predict(X_test_raw)
r2_train_raw = r2_score(y_train_raw, y_train_pred_raw)
r2_raw = r2_score(y_test_raw, y_pred_raw)
print(f"\n--- Training vs Testing Performance ---")
print(f"Training R²: {r2_train_raw:.4f}")
print(f"Testing R²:  {r2_raw:.4f}")

y_pred_raw = model_pipeline_raw.predict(X_test_raw)

# Metrics
r2_raw = r2_score(y_test_raw, y_pred_raw)
mse_raw = mean_squared_error(y_test_raw, y_pred_raw)
rmse_raw_lakhs = np.sqrt(mse_raw) / 1e5
mae_raw = mean_absolute_error(y_test_raw, y_pred_raw)
mae_raw_lakhs = mae_raw / 1e5

# Adjusted R²
n_raw = X_test_raw.shape[0]
p_raw = X_test_raw.shape[1]
adjusted_r2_raw = 1 - (1 - r2_raw) * ((n_raw - 1) / (n_raw - p_raw - 1))

# Prepare results dataframe
y_test_lakhs = y_test_raw / 1e5
y_pred_lakhs = y_pred_raw / 1e5
errors = y_pred_lakhs - y_test_lakhs

results_df = pd.DataFrame({
    'Actual Resale Price (Lakhs)': y_test_lakhs.values,
    'Predicted Resale Price (Lakhs)': y_pred_lakhs,
    'Error (Lakhs)': errors
}, index=y_test_lakhs.index)


results_df['Abs Error'] = results_df['Error (Lakhs)'].abs()
results_df = results_df.sort_values(by='Abs Error', ascending=True)

# Display results
print("\nPer-Car Prediction Results (Top 10 by largest error):")
print(results_df[['Actual Resale Price (Lakhs)', 'Predicted Resale Price (Lakhs)', 'Error (Lakhs)']].head(10))

print("\n--- RAW TARGET RESULTS ---")
print(f"R²: {r2_raw:.4f}")
print(f"Adjusted R²: {adjusted_r2_raw:.4f}")
print(f"MSE: {mse_raw:.2f}")
print(f"RMSE in lakhs: {rmse_raw_lakhs:.2f} lakhs")
print(f"MAE in lakhs: {mae_raw_lakhs:.2f} lakhs")

def plot_feature_importance(pipeline, X_num_columns, X_cat_columns, title):
    # 1️⃣ Get feature names
    preprocessor = pipeline.named_steps['preprocessor']
    
    numeric_features = X_num_columns
    onehot_encoder = preprocessor.named_transformers_['cat']
    onehot_feature_names = onehot_encoder.get_feature_names_out(X_cat_columns)
    
    full_feature_names = numeric_features + list(onehot_feature_names)
    
    # 2️⃣ Get coefficients
    regressor = pipeline.named_steps['regressor']
    coefficients = regressor.coef_
    
    coef_df = pd.DataFrame({
        'Feature': full_feature_names,
        'Coefficient': coefficients
    })
    
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
    
    # 3️Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Coefficient',
        y='Feature',
        data=coef_df.head(25),
        hue='Feature',
        dodge=False,
        palette='coolwarm',
        legend=False
    )
    plt.title(f"Top 25 Feature Importances ({title})")
    plt.xlabel("Coefficient Value (Positive = Increase Resale Price, Negative = Decrease Resale Price)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    # 4️Also print full list if desired:
    print(f"\n===== {title} - Full Feature Coefficients =====")
    print(coef_df[['Feature', 'Coefficient']].to_string(index=False))




# ---------------------------------
# Run it for RAW target model:

plot_feature_importance(
    model_pipeline_raw,
    X_num.columns.tolist(),
    X_cat.columns.tolist(),
    "RAW Target Model"
)





# RAW target plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_raw/1e5, y=y_pred_raw/1e5)
plt.xlabel("Actual Resale Price (Lakhs)")
plt.ylabel("Predicted Resale Price (Lakhs)")
plt.title("RAW Target: Actual vs Predicted Resale Price")
plt.plot([y_test_raw.min()/1e5, y_test_raw.max()/1e5],[y_test_raw.min()/1e5, y_test_raw.max()/1e5], 'r--')
plt.show()



