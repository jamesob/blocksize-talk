import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Define key dates and status
dates = ['1776-07-04', '1933-04-05', '1950-01-01']
status = [1, 0, 0]  # 1=yes, 0=no

# Convert dates to matplotlib format
x = [plt.matplotlib.dates.date2num(datetime.strptime(d, '%Y-%m-%d')) for d in dates]

plt.rcParams.update({'font.size': 18})

# Create the plot
plt.figure(figsize=(4, 3))
plt.step(x, status, where='post', linewidth=2)

# Set y-axis ticks and labels
plt.yticks([0, 1], ['No', 'Yes'])
plt.ylim(-0.5, 1.5)

# Format x-axis with dates
plt.xlim(plt.matplotlib.dates.date2num(datetime.strptime('1776-07-04', '%Y-%m-%d')),
         plt.matplotlib.dates.date2num(datetime.strptime('1950-01-01', '%Y-%m-%d')))
# Create date objects for x-ticks
date_ticks = [datetime.strptime(d, '%Y-%m-%d') for d in
             ['1776-07-01', '1933-04-05']]

# Convert to matplotlib date format
date_ticks_mpl = [plt.matplotlib.dates.date2num(d) for d in date_ticks]

plt.xticks(date_ticks_mpl, fontsize=16)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

# Add title and labels
plt.title("Is it legal to own gold in the United States?")
plt.ylabel("Legal?", fontsize=18)
plt.grid(True, alpha=0.3)

plt.show()
