import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

# Read in steam_vr_users_by_percent.csv
df = pd.read_csv('steam_vr_users_by_percent.csv', parse_dates=[0])

# Plot the data
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df['date'], df['percent']*100, 'ko', label='Data')

# Fit a linear regression model over years using statsmodels
# Convert into years
X = df['date'].dt.year + df['date'].dt.dayofyear / 365
Xmin = int(X.min())
X -= Xmin
y = df['percent'] * 100

# Add a constant term for the intercept
X_sm = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X_sm).fit()

# Print the summary of the regression results
print(model.summary())

# Project 10 years into the future
Xproj = np.arange(Xmin, Xmin+11) - Xmin
Xproj_sm = sm.add_constant(Xproj)
yproj = model.predict(Xproj_sm)

date_new = pd.to_datetime(Xproj+Xmin, format='%Y') # Convert years back to datetime
ax.plot(date_new, yproj, 'r--', label=f"% = {model.params[1]:.3f}*t + {model.params[0]:.3f}\n(percent per year since {df['date'].min().year})")

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Percent of VR Users', fontsize=14)
ax.set_title('Steam VR Users by Percent', fontsize=16)
ax.grid(True, ls='--', alpha=0.5)
ax.legend()

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('steam_vr_users_by_percent.png')
plt.show()