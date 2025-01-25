import matplotlib.pyplot as plt
import numpy as np
import datetime

vr_users = lambda t: 0.000408*t**2 + 0.0562916*t + 0.328348

base_year = 2020
year_since = np.arange(0,11)

users = vr_users(year_since)

# compute the number for today
today = datetime.datetime.now()
days_since = (today - datetime.datetime(base_year,1,1)).days
years_since = days_since/365.25
users_today = vr_users(years_since)

fig,ax = plt.subplots(1, figsize=(7,5))
ax.plot(year_since + base_year, users, 'k--')

# add dot with today and number
ax.plot(today.year, users_today, 'bo')
ax.text(today.year-1, users_today+0.02, f'{users_today:.2f} M', fontsize=12, color='blue')

# add another dot with 2030
users_2030 = vr_users(10)
ax.plot(2030, users_2030, 'go')
ax.text(2028.75, users_2030, f'{users_2030:.2f} M', fontsize=12, color='green')

ax.grid(True, ls='--', alpha=0.5)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Daily Users with VR Hardware (million)', fontsize=14)
ax.set_title('VR User Growth on Steam', fontsize=16)
plt.tight_layout()
plt.savefig('vr_growth_steam.png')
plt.show()