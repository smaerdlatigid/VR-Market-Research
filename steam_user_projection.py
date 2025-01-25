import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and preprocess data
data_file = "steam_chart.csv"
data = pd.read_csv(data_file, parse_dates=[0])
mask = data["DateTime"] >= np.datetime64("2020-01-01")
filtered_data = data[mask].copy()
filtered_data = filtered_data.dropna()

# Simplified time-based feature engineering
def create_time_features(df):
    df = df.copy()
    
    # Basic time features
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['year'] = df['DateTime'].dt.year
    
    # monotonic time feature
    df['time'] = (df['DateTime'] - datetime(2020, 1, 1)).dt.total_seconds()
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/4)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/4)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Weekend indicator
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Time of day categories
    df['time_of_day'] = pd.cut(df['hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    return df

filtered_data = create_time_features(filtered_data)

# Define feature columns
feature_columns = [
    #'hour_sin', 'hour_cos', 
    'day_of_week_sin', 'day_of_week_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'time'
]

# Prepare data
X = filtered_data[feature_columns].values
y = filtered_data['Average Users'].values

# Handle NaN values
nan_mask = ~np.isnan(y)
X = X[nan_mask]
y = y[nan_mask]

# Scale features
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate R-squared scores
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training R² score: {train_score:.4f}")
print(f"Testing R² score: {test_score:.4f}")

# Generate future dates and features
def create_future_features(start_date, end_date, freq='H'):
    future_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    future_df = pd.DataFrame({'DateTime': future_dates})
    return create_time_features(future_df)

future_df = create_future_features('2022-01-01', '2025-12-31')
future_X = future_df[feature_columns].values
future_X_scaled = X_scaler.transform(future_X)

# Generate predictions
future_predictions_scaled = model.predict(future_X_scaled)
future_predictions = y_scaler.inverse_transform(future_predictions_scaled)

# Visualization
fig = plt.figure(figsize=(12, 15))
gs = fig.add_gridspec(3, 2)

# Plot 1: Overall prediction (top row, spans both columns)
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(filtered_data["DateTime"], filtered_data["Average Users"]/1e6, 
            label="Actual Users", color='black')

# smooth the future predictions by taking a rolling average every 24 hours
future_predictions = pd.Series(future_predictions.flatten()).rolling(window=24).mean()

ax_main.plot(future_df["DateTime"], future_predictions/1e6, 
            label="Predicted Users", alpha=0.75, color='red')
ax_main.set_xlim([filtered_data["DateTime"].min(), future_df["DateTime"].max()])
ax_main.set_xlabel("DateTime")
ax_main.set_ylabel("Average Users (Millions)")
ax_main.set_title("Concurrent Steam Users: Actual vs Predicted (Linear Regression)")
ax_main.legend()
ax_main.grid(True)

# Weekly pattern
ax1 = fig.add_subplot(gs[1, 0])
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_data = [filtered_data[filtered_data['day_of_week'] == i]['Average Users']/1e6 for i in range(7)]
ax1.boxplot(weekly_data, tick_labels=days)
ax1.set_title('User Distribution by Day of Week', pad=20)
ax1.set_ylabel('Average Users (Millions)')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_ylim([19,34])

# Monthly pattern
ax2 = fig.add_subplot(gs[1, 1])
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_data = [filtered_data[filtered_data['month'] == i+1]['Average Users']/1e6 for i in range(12)]
#ax2.boxplot(monthly_data, tick_labels=month_names)
# make boxplot more readable
ax2.boxplot(monthly_data, tick_labels=month_names, showfliers=False)
ax2.set_title('User Distribution by Month', pad=20)
ax2.set_ylabel('Average Users (Millions)')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_ylim([19,34])


# Time of day pattern
ax3 = fig.add_subplot(gs[2, 0])
time_periods = ['Night', 'Morning', 'Afternoon', 'Evening']
time_of_day_data = [filtered_data[filtered_data['time_of_day'] == period]['Average Users']/1e6
                    for period in time_periods]
ax3.boxplot(time_of_day_data, tick_labels=time_periods)
ax3.set_title('User Distribution by Time of Day', pad=20)
ax3.set_ylabel('Average Users (Millions)')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_ylim([19,34])

# Yearly trend
ax4 = fig.add_subplot(gs[2, 1])
yearly_avg = filtered_data.groupby('year')['Average Users'].mean()
ax4.plot(yearly_avg.index, yearly_avg.values/1e6, marker='o', linewidth=2, color='black')
# use linalg to fit a line
slope, intercept = np.polyfit(yearly_avg.index - 2020, yearly_avg.values, 1)
# create new year range for prediction
year_range = np.arange(2020, 2031)
ax4.plot(year_range, (slope * (year_range - 2020) + intercept)/1e6,
         linestyle='--', color='red', 
         label=f'y={slope/1e6:.2f}*t+{intercept/1e6:.2f} (Millions since 2020)')
ax4.set_title('Yearly Average Users', pad=20)
ax4.set_ylabel('Average Users (Millions)')
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.savefig('steam_user_projection.png')
plt.show()

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': model.coef_.flatten()
})
print("\nFeature Importance:")
print(feature_importance)

# save the model
model_data = {
    'model': model,
    'X_scaler': X_scaler,
    'y_scaler': y_scaler,
    'feature_columns': feature_columns
}
joblib.dump(model_data, 'steam_user_projection_model.pkl')