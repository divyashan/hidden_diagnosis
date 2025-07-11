import os
import pandas as pd
def restrict_GPU_pytorch(gpuid, use_cpu=False):
    """
        gpuid: str, comma separated list "0" or "0,1" or even "0,1,3"
    """
    if not use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

        print("Using GPU:{}".format(gpuid))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU")


from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

from scipy.stats import binom
import statsmodels.api as sm

import matplotlib.pyplot as plt
import numpy as np
import hashlib
import json


prettify_group_name = {'PatientRaceFinal': 'Race', 'instype_final': 'Insurance', 'PrimLangDSC': 'Primary Language', 'SexDSC': 'Sex', 'instype': 'Insurance Type'}

colors = {'Black or African American': 'black', 'Hispanic or Latino': 'red', 'White': 'blue', 'Asian': 'orange',
          'Medicare': 'black', 'Medicaid': 'red', 'Commercial': 'blue', 
          'Spanish': 'black', 'English': 'blue', 'Other': 'darkorange',
          'lpm_adjusted_risk_score': 'blue', 
          'lr_risk_score_dem_ecg_feats': 'red', 
                                               'lr_risk_score_ecg': 'darkorange', 'lr_risk_score_dems': 'black', 'Male': 'blue', 'Female': 'red',
                                               'Demographics': 'red', 'ECG Preprocessed Features': 'black', 'Our Model': 'blue',
                                               'Non-English': 'black',
                                               }

ecg_feat_names = ['VentricularRate', 'AtrialRate', 'PRInterval',
       'QRSDuration', 'QTInterval', 'QTCorrected', 'PAxis', 'RAxis', 'TAxis',
       'QRSCount', 'QOnset', 'QOffset', 'POnset', 'POffset', 'TOffset',
       'ECGSampleBase']

dem_feat_names = ['binary_Black or African American', 'binary_Hispanic or Latino',
                'binary_Declined or Unavailable','binary_Asian',  'binary_Other', 
                'binary_Native American or Pacific Islander',
                'binary_Female', 
                'PatientAge_years_01', 'binary_Medicare', 'binary_Medicaid', 'binary_NonEnglish']

def deterministic_dict_hash(d):
    """Create a deterministic SHA-256 hash of a dictionary."""
    # Convert the dictionary to a JSON string with sorted keys
    dict_str = json.dumps(d, sort_keys=True, separators=(',', ':'))
    # Encode the string and hash it
    return hashlib.sha256(dict_str.encode('utf-8')).hexdigest()

def prettify_col_name(col_name):
    col_name_map = {'PatientRaceFinal': 'Race', 'SexDSC': 'Sex', 'ecg_location': 'Site', 'PrimLangDSC': 'Primary Language', 'greater_than_60': 'Age > 60', 
                    'greater_than_65': 'Age > 65', 'mortality_within_one_year': 'Mortality < 1 Year', 'hospitalization': 'Hospitalization',
                    'binary_NonEnglish': 'Non-English', 'instype_final': 'Insurance Type', 'diagnosis_in_charts': 'AF Diagnosis', 'hr_over_160': 'ECG with HR > 160', 'stroke_within_year': 'Stroke',
                    'binary_prim_language': 'Primary Language',
                    'label': 'AF ECG (< 90 days)', 'stroke': 'Stroke',
                    'diagnosis_within_year': 'Diagnosis'}
    return col_name_map[col_name]

def calibration_plot(df, n_bins=5, label='Calibration plot', risk_score_col='risk_score', premade_bins=False):
    # Sort by risk score to make binning easier
    df = df.sort_values(by=risk_score_col)
    
    # Calculate bin edges and bin each risk score
    if not premade_bins:
        # bins = np.linspace(0, 1, n_bins + 1)
        # df['bin'] = np.digitize(df[risk_score_col], bins) - 1 --> If you want even width bins within the group. 
        df['bin'] = pd.qcut(df[risk_score_col], q=n_bins, labels=False)

    # Initialize lists for plotting
    bin_centers = []
    mean_predicted_probs = []
    observed_frequencies = []
    y_errors = []
    x_errors = []
    
    # Loop through each bin to calculate values
    for i in range(n_bins):
        bin_data = df[df['bin'] == i]
        
        if bin_data.empty:
            continue
        
        # Calculate bin center for x-axis
        bin_center = bin_data[risk_score_col].mean()
        bin_centers.append(bin_center)
        
        # Calculate mean predicted probability (x value)
        mean_predicted_prob = bin_data[risk_score_col].mean()
        mean_predicted_probs.append(mean_predicted_prob)
        
        # Calculate observed frequency (y value)
        observed_freq = bin_data['label'].mean()
        observed_frequencies.append(observed_freq)
        
        # Calculate y error (binomial proportion confidence interval)
        n = len(bin_data)
        if n > 0:
            # Compute confidence interval using binomial distribution
            y_error = binom.interval(0.95, n, observed_freq)
            # Convert to error bars relative to observed frequency
            y_errors.append([(observed_freq - y_error[0] / n), (y_error[1] / n - observed_freq)])
        else:
            y_errors.append((0, 0))
        
        # Calculate x error (standard deviation of risk scores in the bin)
        x_error = bin_data[risk_score_col].std() / np.sqrt(len(bin_data))
        x_errors.append(x_error)
    
    # Convert lists to numpy arrays for plotting
    mean_predicted_probs = np.array(mean_predicted_probs)
    observed_frequencies = np.array(observed_frequencies)
    y_errors = np.array(y_errors).T  # Transpose for plt.errorbar format
    x_errors = np.array(x_errors)
    
    # Plot calibration curve with error bars
    plt.errorbar(mean_predicted_probs, observed_frequencies, yerr=y_errors, xerr=x_errors,
                 fmt='o', capsize=5, label=label)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True)


def diagnosis_curve_plot(df, grouping, group, n_bins, time_to_diagnosis=None, risk_score_col='risk_score', event_col='diagnosis_in_charts'):
    df_race = df[df[grouping] == group]
    probabilities = df_race[risk_score_col]
    outcomes = df_race[event_col]
    if time_to_diagnosis != None:
        outcomes = df_race['diagnosis_in_charts'] & (df_race['time_to_diagnosis'] <= time_to_diagnosis)

    # Get the calibration curve data
    fraction_of_positives, mean_predicted_value = calibration_curve(outcomes, probabilities, n_bins=n_bins, strategy='quantile')

    quantile_bins = np.percentile(probabilities, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(probabilities, quantile_bins) - 1


    # Initialize arrays to hold bin stats
    bin_counts = np.zeros(n_bins)
    bin_conf_lower = np.zeros(n_bins)
    bin_conf_upper = np.zeros(n_bins)

    # Calculate bin stats
    for i in range(n_bins):
        bin_outcomes = outcomes[bin_indices == i]
        bin_counts[i] = len(bin_outcomes)
        
        # Calculate the binomial confidence intervals
        successes = np.sum(bin_outcomes)
        if bin_counts[i] > 0:
            ci_low, ci_upp = sm.stats.proportion_confint(successes, bin_counts[i], alpha=0.05, method='binom_test')
            bin_conf_lower[i] = ci_low
            bin_conf_upper[i] = ci_upp
        else:
            bin_conf_lower[i] = bin_conf_upper[i] = np.nan

    # Plot the calibration curve with error bars
    plt.errorbar(mean_predicted_value, fraction_of_positives, 
                 yerr=[fraction_of_positives - bin_conf_lower, bin_conf_upper - fraction_of_positives], 
                 fmt='o', capsize=5, label=f'{group}')


def plot_CIs_covariates(CIs_df, crop_plots=True, ax=None, figsize=(6, 6),
                        covariate_names=None, show=True, ylabel_size=12,
                        xlabel_size=12,
                        color_CIs_by_significance=True, fill_between=False, horizontal_lines=True):
    
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    
    CIs_df = CIs_df.replace({'-':np.nan}).dropna()
    if covariate_names is None: covariate_names = CIs_df.index.values
    
    estimates = CIs_df.loc[covariate_names, 'Estimate'].values
    UB = CIs_df.loc[covariate_names, 'Upper bound'].values
    LB = CIs_df.loc[covariate_names, 'Lower bound'].values
    CIs = np.vstack([estimates-LB, UB-estimates])

    #Collect colors:
    if color_CIs_by_significance:
        colors = ['red' if ub < 0 else ('blue' if lb > 0 else 'grey')
                 for ub, lb in list(zip(UB, LB))]
    else:
        colors = ['black']*len(estimates)

    #Plot:
    for estimate, name, CI, color in zip(estimates[::-1],
                                         covariate_names[::-1],
                                         CIs[::, ::-1].T,
                                         colors[::-1]):
        _ = ax.errorbar(x=estimate,
                        y=name,
                        xerr=CI.reshape(2,1),
                        ecolor=color,
                        capsize=5,
                        linestyle='None',
                        linewidth=1.5,
                        marker="D",
                        markersize=5,
                        mfc=color,
                        mec=color)

    ax.tick_params(axis='y', labelsize=ylabel_size)
    ax.tick_params(axis='x', labelsize=xlabel_size)
    #Grids:
    xlim = ax.get_xlim()
    _ = ax.axvline(0, linestyle='--', color='black', alpha=0.75, zorder=-1, linewidth=1.)

    #for val in [0.5, 1, 1.5, 2]:
    #    if xlim[1] > val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)
    #for val in [-0.5, -1, -1.5, -2]:
    #    if xlim[0] < val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)

    #Fill between:
    if fill_between:
        #Home ownership:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), -0.5, 1.5, alpha=.4, color='lightgrey', linewidth=0)
        #Education:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), 6.5, 8.5, alpha=.4, color='lightgrey', linewidth=0)
        #Demographics:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), 10.5, 14.5, alpha=.4, color='lightgrey', linewidth=0)
    if horizontal_lines:
        for y, c in enumerate(colors[::-1]):
            _ = ax.hlines(y=y, xmin=-3, xmax=3, linestyle='--', linewidth=0.5, alpha=0.5, color='grey')
    _ = ax.set_xlim(xlim)
    _ = ax.set_ylim(-0.5, len(estimates)-0.5)
            
    if show: plt.show()
        
    return ax

def plot_CIs_covariates_T(CIs_df, crop_plots=True, ax=None, figsize=(6, 6),
                        covariate_names=None, show=True, ylabel_size=12,
                        xlabel_size=12,
                        color_CIs_by_significance=True, fill_between=False, horizontal_lines=True):
    
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    
    CIs_df = CIs_df.replace({'-':np.nan}).dropna()
    if covariate_names is None: covariate_names = CIs_df.index.values
    
    estimates = CIs_df.loc[covariate_names, 'Estimate'].values
    UB = CIs_df.loc[covariate_names, 'Upper bound'].values
    LB = CIs_df.loc[covariate_names, 'Lower bound'].values
    CIs = np.vstack([estimates-LB, UB-estimates])

    #Collect colors:
    if color_CIs_by_significance:
        colors = ['red' if ub < 0 else ('blue' if lb > 0 else 'grey')
                 for ub, lb in list(zip(UB, LB))]
    else:
        colors = ['black']*len(estimates)

    #Plot:
    for estimate, name, CI, color in zip(estimates[::-1],
                                         covariate_names[::-1],
                                         CIs[::, ::-1].T,
                                         colors[::-1]):
        
        _ = ax.errorbar(x=name,
                        y=estimate,
                        yerr=CI.reshape(2,1),
                        ecolor=color,
                        capsize=5,
                        linestyle='None',
                        linewidth=1.5,
                        marker="D",
                        markersize=5,
                        mfc=color,
                        mec=color)

    ax.tick_params(axis='y', labelsize=ylabel_size)
    ax.tick_params(axis='x', labelsize=xlabel_size)
    #Grids:
    xlim = ax.get_xlim()
    _ = ax.axhline(0, linestyle='--', color='black', alpha=0.75, zorder=-1, linewidth=1.)

    #for val in [0.5, 1, 1.5, 2]:
    #    if xlim[1] > val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)
    #for val in [-0.5, -1, -1.5, -2]:
    #    if xlim[0] < val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)

    #Fill between:
    if fill_between:
        #Home ownership:
        _ = ax.fill_between(np.linspace(ylim[0], ylim[1], 1000, endpoint=True), -0.5, 1.5, alpha=.4, color='lightgrey', linewidth=0)
        #Education:
        _ = ax.fill_between(np.linspace(ylim[0], ylim[1], 1000, endpoint=True), 6.5, 8.5, alpha=.4, color='lightgrey', linewidth=0)
        #Demographics:
        _ = ax.fill_between(np.linspace(ylim[0], ylim[1], 1000, endpoint=True), 10.5, 14.5, alpha=.4, color='lightgrey', linewidth=0)
    if horizontal_lines:
        for y, c in enumerate(colors[::-1]):
            _ = ax.hlines(y=y, xmin=-3, xmax=3, linestyle='--', linewidth=0.5, alpha=0.5, color='grey')
    # _ = a.x.ylim(ylim)
    _ = ax.set_xlim(-0.5, len(estimates)-0.5)
           
    if show: plt.show()
        
    return ax


def plot_auc_with_slices(test_split_df, adjusted_risk_score, target_label, stratify_feature_name, feature_names=None):

    # Compute the ROC and AUC for the entire dataset
    y_true = test_split_df[target_label]
    y_score = test_split_df[adjusted_risk_score]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    overall_auc = auc(fpr, tpr)

    # Plot the overall ROC curve
    plt.plot(fpr, tpr, label=f"Overall (AUC = {overall_auc:.2f})", linewidth=2)

    if feature_names == None:
        feature_names = test_split_df[stratify_feature_name].unique()
    # Compute and plot ROC curves for each unique value in each feature
    for feature_name in feature_names:
                
        subset = test_split_df[test_split_df[stratify_feature_name] == feature_name]

        
        y_true_subset = subset[target_label]
        y_score_subset = subset[adjusted_risk_score]
        fpr_subset, tpr_subset, _ = roc_curve(y_true_subset, y_score_subset)
        auc_subset = auc(fpr_subset, tpr_subset)
        
        # Plot ROC curve for the subset
        plt.plot(
            fpr_subset, 
            tpr_subset, 
            label=f"{prettify_col_name(stratify_feature_name)} = {feature_name} (AUC = {auc_subset:.2f})", 
            linestyle="--"
        )

    # Plot random guess line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)

    # Customize the plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stratified AUROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)


def plot_CIs_covariates(CIs_df, crop_plots=True, ax=None, figsize=(6, 6),
                        covariate_names=None, show=True, ylabel_size=12,
                        xlabel_size=12,
                        color_CIs_by_significance=True, fill_between=False, horizontal_lines=True):
    
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    
    CIs_df = CIs_df.replace({'-':np.nan}).dropna()
    if covariate_names is None: covariate_names = CIs_df.index.values
    
    estimates = CIs_df.loc[covariate_names, 'Estimate'].values
    UB = CIs_df.loc[covariate_names, 'Upper bound'].values
    LB = CIs_df.loc[covariate_names, 'Lower bound'].values
    CIs = np.vstack([estimates-LB, UB-estimates])

    #Collect colors:
    if color_CIs_by_significance:
        colors = ['red' if ub < 0 else ('blue' if lb > 0 else 'grey')
                 for ub, lb in list(zip(UB, LB))]
    else:
        colors = ['black']*len(estimates)

    #Plot:
    for estimate, name, CI, color in zip(estimates[::-1],
                                         covariate_names[::-1],
                                         CIs[::, ::-1].T,
                                         colors[::-1]):
        _ = ax.errorbar(x=estimate,
                        y=name,
                        xerr=CI.reshape(2,1),
                        ecolor=color,
                        capsize=5,
                        linestyle='None',
                        linewidth=1.5,
                        marker="D",
                        markersize=5,
                        mfc=color,
                        mec=color)

    ax.tick_params(axis='y', labelsize=ylabel_size)
    ax.tick_params(axis='x', labelsize=xlabel_size)
    #Grids:
    xlim = ax.get_xlim()
    _ = ax.axvline(0, linestyle='--', color='black', alpha=0.75, zorder=-1, linewidth=1.)

    #for val in [0.5, 1, 1.5, 2]:
    #    if xlim[1] > val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)
    #for val in [-0.5, -1, -1.5, -2]:
    #    if xlim[0] < val: _ = ax.axvline(val, linestyle='-', color='lightgrey', alpha=0.5, zorder=-1, linewidth=1)

    #Fill between:
    if fill_between:
        #Home ownership:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), -0.5, 1.5, alpha=.4, color='lightgrey', linewidth=0)
        #Education:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), 6.5, 8.5, alpha=.4, color='lightgrey', linewidth=0)
        #Demographics:
        _ = ax.fill_between(np.linspace(xlim[0], xlim[1], 1000, endpoint=True), 10.5, 14.5, alpha=.4, color='lightgrey', linewidth=0)
    if horizontal_lines:
        for y, c in enumerate(colors[::-1]):
            _ = ax.hlines(y=y, xmin=-3, xmax=3, linestyle='--', linewidth=0.5, alpha=0.5, color='grey')
    _ = ax.set_xlim(xlim)
    _ = ax.set_ylim(-0.5, len(estimates)-0.5)
            
    if show: plt.show()
        
    return ax


def plot_auc_with_slices(test_split_df, adjusted_risk_score, target_label, stratify_feature_name, feature_names=None):

    # Compute the ROC and AUC for the entire dataset
    y_true = test_split_df[target_label]
    y_score = test_split_df[adjusted_risk_score]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    overall_auc = auc(fpr, tpr)

    # Plot the overall ROC curve
    # plt.plot(fpr, tpr, label=f"Overall (AUC = {overall_auc:.2f})", linewidth=2)

    if feature_names == None:
        feature_names = test_split_df[stratify_feature_name].unique()
    # Compute and plot ROC curves for each unique value in each feature
    for feature_name in feature_names:
                
        subset = test_split_df[test_split_df[stratify_feature_name] == feature_name]

        
        y_true_subset = subset[target_label]
        y_score_subset = subset[adjusted_risk_score]
        fpr_subset, tpr_subset, _ = roc_curve(y_true_subset, y_score_subset)
        auc_subset = auc(fpr_subset, tpr_subset)
        
        # Plot ROC curve for the subset
        plt.plot(
            fpr_subset, 
            tpr_subset, 
            label=f"{prettify_col_name(stratify_feature_name)} = {feature_name} (AUC = {auc_subset:.2f})", 
            linestyle="--"
        )

    # Plot random guess line
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)

    # Customize the plot
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Stratified AUROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
