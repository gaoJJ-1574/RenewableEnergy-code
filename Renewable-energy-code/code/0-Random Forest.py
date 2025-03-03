import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

file_path = '../Experimental data/WF4-66_filled_power.csv'
df = pd.read_csv(file_path)

feature_columns = ['Wind speed - at the height of wheel hub  (m/s)',
                   'Wind speed - at the height of wheel hub (˚)', 'Air temperature  (°C) ', 'Atmosphere (hpa)', 'Relative humidity (%)']
X = df[feature_columns]
y = df['Power (MW)']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importances = rf_model.feature_importances_

feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Meteorological Features')
plt.title('Feature Importance for Power Prediction (Random Forest)')
plt.gca().invert_yaxis()
plt.show()

print(importance_df)
