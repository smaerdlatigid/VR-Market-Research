import numpy as np
import matplotlib.pyplot as plt

# Data
years = np.array([2019, 2020, 2021, 2022, 2023, 2024])  # years
sales = np.array([11.63, 14.02, 16.44, 19.55, 25.82, 34.03])  # cumulative installed based

# Plotting the sales data
plt.figure(figsize=(10, 6))
plt.plot(years, sales, 'o', color='black', label='Actual Base')
plt.title('Total VR Headsets Over Time', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Cumulative Installed Base (in millions)', fontsize=14)
plt.grid(True, ls='--', alpha=0.5)

years_extended = np.arange(2019, 2031)
# Linear model fit
coefficients_linear = np.polyfit(years, sales, 1)
linear_model = np.poly1d(coefficients_linear)
sales_predicted_linear = linear_model(years_extended)

# Calculate R^2 for the linear model
residuals_linear = sales - linear_model(years)
ss_res_linear = np.sum(residuals_linear**2)
ss_tot_linear = np.sum((sales - np.mean(sales))**2)
r_squared_linear = 1 - (ss_res_linear / ss_tot_linear)

# Quadratic model fit
coefficients_quadratic = np.polyfit(years, sales, 2)
quadratic_model = np.poly1d(coefficients_quadratic)
sales_predicted_quadratic = quadratic_model(years_extended)

# Calculate R^2 for the quadratic model
residuals_quadratic = sales - quadratic_model(years)
ss_res_quadratic = np.sum(residuals_quadratic**2)
ss_tot_quadratic = np.sum((sales - np.mean(sales))**2)
r_squared_quadratic = 1 - (ss_res_quadratic / ss_tot_quadratic)

# Plotting the models
plt.plot(years_extended, sales_predicted_linear, label=f'Linear Model ($R^2$={r_squared_linear:.2f})')
plt.plot(years_extended, sales_predicted_quadratic, label=f'Quadratic Model ($R^2$={r_squared_quadratic:.2f})')

# Projected sales for 2030
projected_sales_2030_linear = linear_model(2030)
projected_sales_2030_quadratic = quadratic_model(2030)

plt.legend()

# add text for each point
for i, txt in enumerate(sales):
    plt.text(years[i]-0.1, sales[i]+0.1, f'{sales[i]}M', verticalalignment='bottom', horizontalalignment='right')

# add text for projected sales in 2030
plt.text(2030-0.1, projected_sales_2030_linear, f'Projected Base in 2030\n (Linear): {projected_sales_2030_linear:.2f}M',
            verticalalignment='bottom', horizontalalignment='right')
plt.text(2030-0.25, projected_sales_2030_quadratic-4, f'Projected Base in 2030\n (Quadratic): {projected_sales_2030_quadratic:.2f}M',
            verticalalignment='bottom', horizontalalignment='right')

plt.xlim([2018, 2030])
plt.tight_layout()
plt.savefig('headset_base_projection.png')
plt.show()