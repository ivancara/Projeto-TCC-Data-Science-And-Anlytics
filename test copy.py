#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
df = pd.read_csv('data/files/Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv', sep=',')


#%%
# split time column into month and year
df['time'] = pd.to_datetime(df['time'])
df['start.time'] = pd.to_datetime(df['start.time'])
df['total.period'] = (df['time'] - df['start.time']).dt.days
df['sex'].fillna('unknown', inplace=True)
df = pd.get_dummies(df, columns=['sex'], dtype=np.int64)
df['period.name'] = df['period.name'].replace('evening', 0)
df['period.name'] = df['period.name'].replace('midday', 1)
df['period.name'] = df['period.name'].replace('morning', 2)
df['depression_severity'] = df[['phq1', 'phq2', 'phq3','phq4', 'phq5', 'phq6','phq7', 'phq8', 'phq9']].sum(axis=1)
df = df.drop(columns = ['phq1', 'phq2', 'phq3','phq4', 'phq5', 'phq6','phq7', 'phq8', 'phq9'])
df = df.drop(columns = ['q1', 'q2', 'q3','q4', 'q5', 'q6','q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q16', 'q46', 'q47'])
df = df.drop(columns = ['Unnamed: 0','time', 'start.time', 'user_id', 'period.name', 'phq.day'])
df = df[~df['age'].isnull()]
df = df.drop('id', axis = 1)
df = df[df['sex_unknown'] == 0]
df = df.drop('sex_unknown', axis=1)
df = df[df['sex_transgender'] == 0]
df = df.drop('sex_transgender', axis=1)
scaler = StandardScaler()
scaler.fit(df)
scaled_df = pd.DataFrame(scaler.transform(df), columns = df.columns)

#%%
correlation = df.corr(method='pearson')

mask = np.triu(np.ones_like(correlation))
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), cmap = 'viridis', mask = mask,
            annot = True, fmt='.1f')

plt.show()

# sx = 
# sns.heatmap(correlation, cmap='coolwarm', mask=mask,vmax=1,square=True, linewidths=.5)
# plt.show()

# %%
features = df[['age', 'total.period', 'sex_female', 'sex_male', 'depression_severity']]
target = df['happiness.score']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)
# %%
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared (R2)
r2 = r2_score(y_test, y_pred)
print("R-squared (R2):", r2)
# %%
