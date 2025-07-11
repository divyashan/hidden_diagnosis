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
    for estimate, name, CI, color in zip(estimates[::-1], covariate_names[::-1], 
                                         CIs[::, ::-1].T, colors[::-1]):
        
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
