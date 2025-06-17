# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:20:44 2025
Code to plot the Mastersizer data with no fittings
@author: vb22224
"""

import pandas as pd
import matplotlib.pyplot as plt



file_name = r"C:\Users\vb22224\OneDrive - University of Bristol\Desktop\MAIN\Data for Electrostatics Paper\Mastersizer Data\Data.csv"

plt.figure(dpi=600)

# Get x-axis data (Size column)
df = pd.read_csv(file_name)
x = df['Size']

# Plot each column except 'Size' as y data
for column in df.columns:
    if column != 'Size':
        plt.plot(x, df[column], label=column)

# Customize the plot
plt.xlabel('Particle Size / μm')
plt.ylabel('Volume Frequency Density / %')
plt.legend()
plt.xscale('log')
plt.xlim([0.1,3000])
plt.ylim([0, 13])
# Display the plot
plt.show()