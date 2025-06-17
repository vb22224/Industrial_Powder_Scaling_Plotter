# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:45:30 2024
Code to plot charge change against mass, with varying fitting options
@author: vb22224
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, Normalize
from collections import defaultdict
import warnings



def get_fit(x, ft='linear', popt=[0,0,0], snl=2/3, force_0=True):

    """ Gets the predicted fitted y values """
        
    if ft == 'linear':
        y_pred = popt[0] * x
    elif ft == 'non-linear':
        y_pred = popt[0] * x**(2/3)
    elif ft == 'set-non-linear':
        y_pred = popt[0] * x**snl
    elif ft == 'free-non-linear':
        y_pred = popt[0] * x**popt[1]   
    
    if not force_0:
        y_pred += popt[-1]
            
    return y_pred

    

def get_vminmax(df, config):
    
    """ Calculate global min and max for color scaling """
    
    if config['colour_by_self']:
        all_self_charge = []
        for _, group in df.groupby('category'):
            self_charge = (group['charge_MM'] - np.abs(group['charge'])) * group['masses']
            all_self_charge.extend(self_charge.dropna().tolist())
        vmin, vmax = min(all_self_charge), max(all_self_charge)
    else:
        vmin, vmax = None, None
        
    return vmin, vmax



def print_average_parameters():
    
    """ Function to calculate the overall fitting parameters """
    
    print("\nAverage Fitting Parameters:")
    print("-" * 50)
    
    # Calculate averages and standard deviations
    all_values = defaultdict(list)
    for category, params in fitting_params.items():
        if category not in config['skip_cat']:
            for i, param in enumerate(params):
                all_values[param_names[i]].append(param)
    
    # Print the results
    for param_name in param_names:
        values = all_values[param_name]
        if values:
            avg = np.mean(values)
            std = np.std(values)
            if avg != 0 or std != 0:
                print(f"{param_name}: {avg:.2f} ± {std:.2f}")
                
    return



def analyze_spread(mass, charge, window_size=10):
    
    """ Analyzes the spread of charge data as a function of mass using a moving window approach """

    # Convert numpy arrays to pandas Series if they aren't already
    if isinstance(mass, np.ndarray):
        mass = pd.Series(mass)
    if isinstance(charge, np.ndarray):
        charge = pd.Series(charge)
    
    sorted_indices = np.argsort(mass) # Sort data by mass and keep track of original indices
    
    # Reset indices and use iloc to ensure proper integer-based indexing of pandas Series
    mass = mass.reset_index(drop=True)
    charge = charge.reset_index(drop=True)
    sorted_mass = mass.iloc[sorted_indices]
    sorted_charge = charge.iloc[sorted_indices]
    
    # Initialize arrays for results
    n_windows = len(mass) - window_size + 1
    mass_centers = np.zeros(n_windows)
    spreads = np.zeros(n_windows)
    window_indices = []  # Store indices for each window
    
    # Calculate spread in each window
    for i in range(n_windows):
        window_mass = sorted_mass[i:i+window_size]
        window_charge = sorted_charge[i:i+window_size]
        window_idx = sorted_indices[i:i+window_size]  # Original indices for this window
        
        mass_centers[i] = np.mean(window_mass)
        spreads[i] = np.std(window_charge)
        window_indices.append(window_idx)
    
    return mass_centers, spreads, window_indices



def calc_plot_fit(mass, charge, legend_handles, config, label='Overall', color='#1f77b4', marker='o'):
    
    """ Calculates and plots the fits for the data """
    
    ft, se, snl, f0, wv = config['fit_type'], config['standard_errors'], config['set_non_linear'], config['force_0'], config['weight_variance'] # Extract commonly used parameters
    if ft != 'linear' and ft != 'non-linear' and ft != 'free-non-linear' and ft != 'set-non-linear': # Checking for valid fit type
        raise ValueError(f'The fit_type should be: linear, non-linear, free-non-linear, or none. Instead given: {ft}')
    
    if f0:
        if ft == 'linear': # Linear fitting
            popt = [np.sum(mass * charge) / np.sum(mass ** 2), 0]
        elif ft == 'non-linear':
            def fit_func(x, a):
                return a * x**(2/3)
        elif ft == 'set-non-linear':
            def fit_func(x, a):
                return a * x**snl
        elif ft == 'free-non-linear':
            def fit_func(x, a, b):
                return a * x**(b)
    else:  
        if ft == 'linear': # Standard linear regression
            popt = np.polyfit(mass, charge, 1)
        elif ft == 'non-linear':
            def fit_func(x, a, c):
                return a * x**(2/3) + c
        elif ft == 'set-non-linear':
            def fit_func(x, a, c):
                return a * x**snl + c
        elif ft == 'free-non-linear':
            def fit_func(x, a, b, c):
                return a * x**(b) + c
            
    # small_data = pd.read_csv("C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Polypropylene Thiovit Flour.csv", sep=',')
    # small_data = small_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['masses', config["charge_type"]])
    # filtered_data = small_data[(small_data['category'] == 'Polypropylene') & (small_data['masses'] < 1)]
    # small_mass = filtered_data['masses']
    # small_specific_charge = filtered_data[config["charge_type"]]
    # small_charge = small_specific_charge * small_mass
    # mass = pd.concat([mass, small_mass])
    # charge = pd.concat([charge, small_charge])
    
    if wv == 'none':
        sigma = None
    elif wv == 'SDV':
        mass_centers, spreads, _ = analyze_spread(mass, charge, window_size=10)
        spread_interpolator = interp1d(mass_centers, spreads, kind='linear', fill_value='extrapolate')
        sigma = spread_interpolator(mass)
    elif wv == 'R2':
        sigma = np.sqrt(421.71*mass**2 + 94626*mass + 404941) # Standard deviation of residuals (sqrt handles that the fitting is for residuals^2)
    else:
        raise ValueError(f'weight_variance should be residuals squared (R2), Sliding Door Variance (SDV), or not at all (none), instead given: {wv}')
    
    if ft != 'linear': # Performing the non-linear fitting
        
        if ft == 'free-non-linear':
            a, b = config['free_non_linear_guess'][:2]
            if f0:
                p0 = [a, b]  # Initial guesses for a, b
            else:
                p0 = [a, b, 0.0]  # Initial guesses for a, b, c
            popt, pcov = curve_fit(fit_func, mass, charge, p0=p0, 
                                 sigma=sigma, absolute_sigma=True)
        else: 
            popt, pcov = curve_fit(fit_func, mass, charge, 
                                 sigma=sigma, absolute_sigma=True)
            a_fit, *other_params = popt  # Get the first parameter (for non-linear, this is the "a" term)
            a_err, *other_param_errors = np.sqrt(np.diag(pcov))  # Get the standard errors for all parameters
        
    y_pred = get_fit(mass, ft, popt, snl, f0) # Getting the predicted y-values
    
    if not config['plot_together']: # Printing which catergory is being calculated
        print(f'\n{cat}:')
    
    # Calculate and print the R² value of the fit
    residuals = charge - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((charge - np.mean(charge))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R\u00B2 value of the fit: {r_squared:.2f}")
        
    if config['plot_residuals']:
        if config['square_residuals']: residuals *= residuals
        fig2 = plt.figure(dpi=config['dpi'])
        ax2 = fig2.add_subplot(111)
        ax2.scatter(mass, residuals, alpha=0.7, color=color)
        
        # Fit polynomial to residuals
        poly_degree = 2  # Adjust degree as needed
        poly_coeff = np.polyfit(mass, residuals, poly_degree)
        mass_fine = np.linspace(min(mass), max(mass), 1000)  # Smooth x-axis points
        residuals_fit = np.polyval(poly_coeff, mass_fine)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Mass / g')
        
        if config['square_residuals']:
            ax2.set_ylabel('Residuals Squared / pC$^2$')
            ax2.plot(mass_fine, residuals_fit, 'r-', alpha=0.8, label=f'Polynomial fit (degree {poly_degree})') # Plot the polynomial fit
            print("\nPolynomial fit y = ax² + bx + c")
            print(f'a = {poly_coeff[0]}, b = {poly_coeff[1]}, c = {poly_coeff[2]}')
        else:
            ax2.set_ylabel('Residuals / pC')
            
        std_residuals = np.std(residuals)
        ax2.axhline(y=config['standard_errors']*std_residuals, color='gray', linestyle='--', alpha=0.7)
        ax2.axhline(y=-config['standard_errors']*std_residuals, color='gray', linestyle='--', alpha=0.7)
        ax2.legend()
        plt.figure(fig1.number)

    # Custom range of values to fit best fit line over
    x_fit_0_to_p2 = np.linspace(0, 0.02, 10)
    x_fit_0_to_2 = np.linspace(0, 2.0, 100)
    x_fit_0_to_200 = np.linspace(0, 200, 100)
    x_fit_200_to_large = np.linspace(200, 1000000, 200)
    x_fit = np.sort(np.concatenate([x_fit_0_to_2, x_fit_0_to_p2, x_fit_0_to_200, x_fit_200_to_large]))
    y_fit = get_fit(x_fit,ft, popt,snl, f0) # getting the fit
    plt.plot([0, 420000], [0, 0], '--', linewidth=1.0, color='black') 
    plt.plot(x_fit, y_fit, '--', color=color)  # Plot the best-fit line

    # Calculating the errors and printing/plotting them as required
    if ft == 'linear':
        n = len(mass)
        ser = np.sqrt(ss_res / (n - 2))  # Standard error of the regression
        x_mean = np.mean(mass)
        ss_xx = np.sum((mass - x_mean)**2)
    
        if f0:
            slope_error = ser / np.sqrt(np.sum(mass**2))
            intercept_error = 0
        else:
            slope_error = ser / np.sqrt(ss_xx)
            intercept_error = ser * np.sqrt(1/n + (x_mean**2 / ss_xx))

        y_upper = (popt[0] + slope_error * se) * x_fit + (popt[1] + intercept_error * se)
        y_lower = (popt[0] - slope_error * se) * x_fit + (popt[1] - intercept_error * se)
        print(f"Slope = {popt[0]:.2f} ± {slope_error:.2f}")
        if not f0:
            print(f"Intercept = {popt[1]:.2f} ± {intercept_error:.2f}")
        
    elif ft == 'non-linear':
        if f0:
            a_fit = popt[0]
            a_err = np.sqrt(np.diag(pcov))[0]  # Extract the first standard error (for 'a')
            y_upper = a_fit * x_fit**(2/3) + a_err * se
            y_lower = a_fit * x_fit**(2/3) - a_err * se
            print(f"a = {a_fit:.2f} ± {a_err:.2f}")
        else:
            a_fit, c_fit = popt  # a and c parameters for non-linear fit
            a_err, c_err = np.sqrt(np.diag(pcov))[:2]  # Standard errors for a and c
            y_upper = (a_fit * x_fit**(2/3) + c_fit) + np.sqrt((x_fit**(2/3) * a_err)**2 + c_err**2) * se
            y_lower = (a_fit * x_fit**(2/3) + c_fit) - np.sqrt((x_fit**(2/3) * a_err)**2 + c_err**2) * se
            print(f"a = {a_fit:.2f} ± {a_err:.2f}")
            print(f"c = {c_fit:.2f} ± {c_err:.2f}")
    
    elif ft == 'set-non-linear':
        if f0:
            a_fit = popt[0]
            a_err = np.sqrt(np.diag(pcov))[0]  # Extract the first standard error (for 'a')
            y_upper = a_fit * x_fit**snl + a_err * se
            y_lower = a_fit * x_fit**snl - a_err * se
            print(f"a = {a_fit:.2f} ± {a_err:.2f}")
        else:
            a_fit, c_fit = popt  # a and c parameters for non-linear fit
            a_err, c_err = np.sqrt(np.diag(pcov))[:2]  # Standard errors for a and c
            y_upper = (a_fit * x_fit**snl + c_fit) + np.sqrt((x_fit**snl * a_err)**2 + c_err**2) * se
            y_lower = (a_fit * x_fit**snl + c_fit) - np.sqrt((x_fit**snl * a_err)**2 + c_err**2) * se
            print(f"a = {a_fit:.2f} ± {a_err:.2f}")
            print(f"c = {c_fit:.2f} ± {c_err:.2f}")
        
    elif ft == 'free-non-linear':
        if f0:
            a_fit, b_fit = popt[:2]
            a_err, b_err = np.sqrt(np.diag(pcov))[:2]
            y_upper = (a_fit * x_fit**b_fit) + np.sqrt((x_fit**b_fit * a_err)**2 + (a_fit * x_fit**b_fit * np.log(x_fit) * b_err)**2) * se
            y_lower = (a_fit * x_fit**b_fit) - np.sqrt((x_fit**b_fit * a_err)**2 + (a_fit * x_fit**b_fit * np.log(x_fit) * b_err)**2) * se
            print(f"a = {a_fit:.2f} ± {a_err:.2f}")
            print(f"b = {b_fit:.2f} ± {b_err:.2f}")
        else:
            a_fit, b_fit, c_fit = popt  # a, b, and c parameters for free non-linear fit
            a_err, b_err, c_err = np.sqrt(np.diag(pcov))[:3]  # Standard errors for a, b, and c
            y_upper = (a_fit * x_fit**b_fit + c_fit) + np.sqrt((x_fit**b_fit * a_err)**2 + (a_fit * x_fit**b_fit * np.log(x_fit) * b_err)**2 + c_err**2) * se
            y_lower = (a_fit * x_fit**b_fit + c_fit) - np.sqrt((x_fit**b_fit * a_err)**2 + (a_fit * x_fit**b_fit * np.log(x_fit) * b_err)**2 + c_err**2) * se
            print(f"a = {a_fit:.2f} ± {a_err:.2f}")
            print(f"b = {b_fit:.2f} ± {b_err:.2f}")
            print(f"c = {c_fit:.2f} ± {c_err:.2f}")
    
    if f0: # Storing fitting parameters
        if ft == 'linear':
            fitting_params[label].extend([popt[0], 1.0, 0.0])  # slope, power=1, intercept=0
        elif ft == 'non-linear':
            fitting_params[label].extend([a_fit, 2/3, 0.0])
        elif ft == 'set-non-linear':
            fitting_params[label].extend([a_fit, snl, 0.0])
        elif ft == 'free-non-linear':
            fitting_params[label].extend([a_fit, b_fit, 0.0])
    else:
        if ft == 'linear':
            fitting_params[label].extend([popt[0], 1.0, popt[1]])  # slope, power=1, intercept
        elif ft == 'non-linear':
            fitting_params[label].extend([a_fit, 2/3, c_fit])
        elif ft == 'set-non-linear':
            fitting_params[label].extend([a_fit, snl, c_fit])
        elif ft == 'free-non-linear':
            fitting_params[label].extend([a_fit, b_fit, c_fit])
            
    fitting_params[label].append(r_squared)
    
    if config['error_area']:  # Fill the uncertainty band
        plt.fill_between(x_fit, y_lower, y_upper, color=color, alpha=0.2)
    
    if config['display_equation']:
        if ft == 'linear':
            full_label = f'{label}: y = {popt[0]:.1f}x'
        elif ft == 'non-linear':
            full_label = f'{label}: y = {popt[0]:.1f}x²ᐟ³'
        elif ft == 'set-non-linear':
            full_label = f'{label}: y = {popt[0]:.1f}x^{snl:.2f}'
        elif ft == 'free-non-linear':
            full_label = f'{label}: y = {popt[0]:.1f}x^{popt[1]:.2f}'
                
        if not f0:
            sign = '+' if popt[-1] >= 0 else '-'
            full_label +=  f' {sign} {abs(popt[-1]):.1f}'
    else:
        full_label = label
    
    marker_color = 'gray' if config['colour_by_self'] else color
    
    legend_handles.append(Line2D([0], [0], marker=marker, color=color, markerfacecolor=marker_color, 
                                 markeredgecolor=marker_color, label=full_label, linestyle='--', linewidth=1))
    
    return legend_handles
                


def fit_data(data, legend_handles, label, color, config, marker='o'):
    
    """ Calculating the charge, plotting the data and calling the function to calculate and plot the fit """
    
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=['masses', config["charge_type"]])
    # data = data[data["masses"] > 2]
    
    if data.empty:
        print(f"Skipping {label}: no valid data after dropping NaNs/Infs.")
        return legend_handles
    
    mass = data['masses']
    specific_charge = data[config["charge_type"]] 
    # if label == 'Thiovit': specific_charge = -data['charge_MM'] # If treating Thiovit differently
    charge = specific_charge * mass
    
    if config['charge_mag']:
        charge = np.abs(charge)
    
    scatter_color = marker_color = color
    if config['colour_by_self']:
        scatter_color = (data['charge_MM'] - np.abs(data['charge'])) * mass
        vmin, vmax = get_vminmax(df, config)
        marker_color = 'gray'
        
        if config['colour_op'] == 'linear':
            colour_op = Normalize(vmin=vmin, vmax=vmax)
        elif config['colour_op'] == 'log':
            colour_op = LogNorm(vmin=vmin, vmax=vmax)
        else:
            raise ValueError(f'colour_op should be linear or log, instead given: {config["colour_op"]}')
    else:
        colour_op = None

    scatter = plt.scatter(mass, charge, c=scatter_color, cmap='viridis', norm=colour_op, marker=marker, alpha=config['alpha'])

    if config['fit_type'] != 'none' and label not in config['skip_cat']:
        legend_handles = calc_plot_fit(mass, charge, legend_handles, config, label, color, marker)
    else:
        legend_handles.append(plt.scatter([], [], color=marker_color, marker=marker, label=label, alpha=config['alpha']))
        
    return legend_handles, scatter, mass, charge



if __name__ == "__main__":

    config = {
        # "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Overall Scaling Comparison.csv",
        # "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Overall Polypropylene Scaling Comparison.csv",
        # "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Polypropylene small vs medium.csv",
        # "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Medium_Scale_Polypropylene/Material Scaling Comparison.csv",
        # "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Thiovit Scaling Comparison Unconditioned.csv",
        # "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Flour Scaling Comparison.csv",
        # "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Thiovit Conditioned vs Unconditioned.csv",
        "file_path": "C:/Users/vb22224/OneDrive - University of Bristol/Desktop/MAIN/Drops/Fume_Hood_Tests/Polypropylene Thiovit Flour.csv",
        "color_list": ["#1f77b4", "#ff7f0e", "c", "#2ca02c", "m", "y", "darkgoldenrod", "yellowgreen", "indigo", "r", "g", "b"],
        "fit_type":'free-non-linear',          # The type of fit to make: linear (y = mx +c), non-linear (y = ax^2/3 + c), free-non-linear (y = ax^b + c), set-non-linear (y = ax^*set* + c) or none
        "force_0": True,             # Whether to froce the line of best fit throgh the origin
        "plot_together": False,        # If True then plots together, but if false categorises them
        "group_by":'category',        # If plot_together = False, which parameter to group by (e.g. 'category' or 'pipe_length')
        "display_equation": False,     # Whether to display the equation for the line of best fit on the graph or not
        "print_av_stats": True,       # Whether to print the averarage fitting paramters bewteen diffent groups
        "error_area": False,           # Whether to plot the area of uncertainty of the fit
        "charge_mag": False,           # Use charge magnintude if True
        "colour_by_self": False,       # If True colours the points by rough self-charging (MM - SE)
        "shape_by_cat": True,         # If True plots each category a different shape (only works if plot_together = False)
        "analyse_spread":False,        # Whether to also analyse the spread of the data (will treat te data all togther)
        "plot_residuals":False,        # Whether to plot the residuals of the fit
        "square_residuals":False,        # Whether to square the residuals if plotted
        "weight_variance":'SDV',     # How to weight the fitting by varince using the residuals squared (R2), Sliding Door Variance (SDV), or not at all (none)
        "colour_op":'linear',         # Scaling option for the colourba to be linear or log
        "standard_errors": 3,         # Number of standard errors (in propgated atandard deviations) for the error region
        "set_non_linear": 2/3,        # Exponent if fit_type = 'set_non_linear'
        "free_non_linear_guess": [-500, 2/3], # Initial guess if fit_type = 'free_non_linear', in the form [a, b], where y = ax^b (+c)
        "alpha":0.5,                  # The transparemcy of the datapoints (1 = solid, 0 = completely see through)
        "charge_type": 'charge_MM',      # 'charge' is SE, can also use 'charge_MM', or 'charge_EE'
        "skip_cat": ['Carbon Black', 'Polypropylene - FH', 'Cellulose', 'Cellulose', 'Thiovit_Medium_Re-unconditioned', 'Thiovit_Medium_Unconditioned', 'Thiovit_Medium_Conditioned'], # Names of categories to not do fittings for
        "together_label": 'Polypropylene',
        "dpi": 600
    }

    # Errors to ignore
    warnings.filterwarnings('ignore', message='No data for colormapping provided via.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', message='divide by zero encountered in log')
    warnings.filterwarnings('ignore', message='invalid value encountered in multiply')
    
    # Setting global parameters
    df = pd.read_csv(config['file_path'], sep=',')
    # df = df[df["fume_hood"] == False] # Only show non-fumehood runs
    # df = df[df['pipe_length'] == 0.3725] # Only show small scale runs
    fig1 = plt.figure(dpi=config['dpi'])
    fitting_params = defaultdict(list)
    param_names = ['a', 'b', 'c', 'R²']
    legend_handles, masses, charges, markers = [], [], [], ['o']
    if config['shape_by_cat']: markers =['s', 'D', '^', 'v', '>', '<', 'p', 'h', '8']  # Add more if needed
    
    if config['plot_together']:
        color = config['color_list'][0]
        legend_handles, scatter, masses, charges = fit_data(df, legend_handles, config['together_label'], color, config)
    else:
        for i, (cat, group) in enumerate(df.groupby(config['group_by'])):
            if 'polypropylene' in str(cat).lower():
                color = '#1f77b4'  # Blue for polypropylene
                marker = 'o'       # Circle for polypropylene
            else:
                color = config['color_list'][(i+1) % len(config['color_list'])]  # Skip blue for others
                marker = markers[i % len(markers)]
            legend_handles, scatter, cat_mass, cat_charge = fit_data(group, legend_handles, cat, color, config, marker=marker)
            masses.extend(cat_mass)
            charges.extend(cat_charge)

    # plt.errorbar(400000, -9000000, yerr=2000000, fmt='o', color='black', ecolor='black', capsize=5) # 2015 Drops Averaged
    # plt.errorbar(400000, 9000000, yerr=2000000, fmt='o', color='black', ecolor='black', capsize=5) # 2015 Drop magnitude
    # legend_handles.append(Line2D([0], [0], marker='o', color='black', label='Polypropylene: 2015', markersize=8, linestyle='None'))
    plt.scatter([400000, 400000, 400000, 400000, 400000, 400000, 400000], # 2015 (400 kg) drops seperately
    [19200000, 12200000, 2200000, 9180000, 3780000, 8840000, 8320000],
    marker='o', facecolors='none', color='#1f77b4', alpha=config['alpha'])
    legend_handles.insert(-1, Line2D([0], [0], marker='o', color='#1f77b4', markerfacecolor='none',label='Polypropylene (400 kg)', linestyle='None'))
    
    
    plt.xlabel('Mass / g')
    # plt.ylabel('Total Charge / pC')
    plt.ylabel('Charge Magnitude / pC')
    # plt.xlim([0, 2.0]) # Very small scale (Thiovit)
    # plt.ylim([0, 150])
    # plt.xlim([0, 1.0]) # Small scale (Flour)
    # plt.ylim([0, 2000])
    # plt.xlim([0, 1.0]) # Small scale (neg)
    # plt.ylim([-1200, 100])
    # plt.xlim([0, 2.0]) # Small scale
    # plt.ylim([-1500, 2000])
    # plt.xlim([0, 200]) # Medium scale
    # plt.ylim([-40000, 1000])
    # plt.xlim([0, 200]) # Medium scale (Thiovit)
    # plt.ylim([0, 10000])
    # plt.xlim([0, 40]) # Medium scale (Flour)
    # plt.ylim([0, 25000])
    # plt.xlim([0.1, 200]) # Medium scale (log)
    # plt.ylim([1, 100000])
    # plt.xlim([0, 420000]) # Large scale
    # plt.ylim([-12000000, 0])
    plt.xlim([0.1, 1000000]) # Large scale (log)
    plt.ylim([1, 100000000])
    plt.xscale('log')
    plt.yscale('log')
    
    plt.legend(handles=legend_handles) # , bbox_to_anchor=(1.6, 1)
    if config['colour_by_self']: plt.colorbar(scatter, label='Self-charge / pC')
    save_name_png = os.path.splitext(config['file_path'])[0] + "_mass_vs_charge.png"
    save_name_pdf = os.path.splitext(config['file_path'])[0] + "_mass_vs_charge.pdf"
    plt.savefig(fname=save_name_png, format='png', bbox_inches='tight', dpi=config['dpi'])  # Save as PNG
    plt.savefig(fname=save_name_pdf, format='pdf', bbox_inches='tight', dpi=config['dpi'])  # Save as PDF
    plt.show()
    
    
    if config['print_av_stats'] and not config['plot_together']: print_average_parameters()
    
    if config['analyse_spread']:
        masses_array = np.array(masses)
        charges_array = np.array(charges)
        plt.figure(dpi=config['dpi'])

        mass_centers, spreads, window_indices = analyze_spread(masses_array, charges_array)
        
        if config['plot_together']:
            plt.scatter(mass_centers, spreads, color='grey', alpha=config['alpha'])
        else:
            legend_elements = []
            categories = sorted(df[config['group_by']].unique(), reverse=True)  # Sort categories in reverse
            
            for cat in categories:
                color_idx = list(categories).index(cat)
                color = config['color_list'][color_idx % len(config['color_list'])]
                legend_elements.append(plt.scatter([], [], color=color, alpha=config['alpha'], label=f"{cat}m"))
            
            for i in range(len(mass_centers)):
                window_data = df.iloc[window_indices[i]]
                category_counts = window_data[config['group_by']].value_counts()
                most_common_cat = category_counts.index[0]
                
                color_idx = list(categories).index(most_common_cat)
                color = config['color_list'][color_idx % len(config['color_list'])]
                
                plt.scatter(mass_centers[i], spreads[i], color=color, alpha=config['alpha'])
            
            plt.legend(handles=legend_elements)
        
        plt.xlabel('Mass / g')
        plt.ylabel('Standard Deviation in Charge / pC')
        plt.show()
        
        
        
