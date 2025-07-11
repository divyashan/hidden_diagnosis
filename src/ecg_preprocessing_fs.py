import numpy as np
import os
import scipy
import pandas as pd
import time
import hashlib 

def baseline_wander_removal(data, sampling_frequency):
    row,__ = data.shape
    processed_data = np.zeros(data.shape)
    for lead in range(0,row):
        # Baseline estimation
        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(data[lead,:], win_size)
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(baseline, win_size)
        # Removing baseline
        filt_data = data[lead,:] - baseline
        processed_data[lead,:] = filt_data
    return processed_data

def ecg_remove_baseline_wander_batch(paths, output_dir='./processed_data/ekg_bwr'):
    os.makedirs(output_dir, exist_ok=True)
    dataset = np.stack([np.load(path) for path in paths], axis=0)

    time1 = time.time()
    output = []
    if dataset.shape[1:] == (2500,12,1):
        dataset = np.transpose(dataset, axes=[0,2,1,3])
    elif dataset.shape[1:] == (1,2500,12):
        dataset = np.transpose(dataset, axes=[0,3,2,1])
    elif dataset.shape[1:] == (12,2500,1):
        pass
    
    for ecg in dataset:
        assert ecg.shape == (12,2500,1), "ecg is not 12,2500"
        processed_data = baseline_wander_removal(ecg.squeeze(),250)
        output.append(processed_data)
        
    bwr_ecg_paths = []
    for i in range(len(output)):
        original_path = paths[i].split('/')[-1]
        bwr_ecg_path = output_dir + '/' + original_path
        bwr_ecg_paths.append(bwr_ecg_path)
        np.save(bwr_ecg_path, output[i])
    return bwr_ecg_paths

BWR_OUTPUT_DIR = './processed_data/ekg_bwr'
os.makedirs(BWR_OUTPUT_DIR, exist_ok=True)

def ecg_remove_baseline_wander_path(path):
    ecg = np.load(path) 

    if ecg.shape == (2500,12,1):
        ecg = np.transpose(ecg, axes=[1,0,2])
    elif ecg.shape == (1,2500,12):
        ecg = np.transpose(ecg, axes=[2,1,0])

    
    assert ecg.shape == (12,2500,1), "ecg is not 12,2500"
    processed_data = baseline_wander_removal(ecg.squeeze(),250)
    
    original_path = path.split('/')[-1]
    bwr_ecg_path = BWR_OUTPUT_DIR + '/' + original_path
    np.save(bwr_ecg_path, processed_data)
    return bwr_ecg_path


def truncate_data_percentile(target_array, lowerbound, upperbound):
    datapctlimit = target_array
    assert (len(upperbound) == datapctlimit.shape[1]), "shape of upperbound does not match array.shape[1]"
    # Truncate data to 0.1th and 99.9th percentile
    for i in range(len(upperbound)):
        print("pre-truncation min:",datapctlimit[:, i, :, :].min())
        print("pre-truncation max:", datapctlimit[:, i, :, :].max())
        datapctlimit[:, i, :, :] = np.where(datapctlimit[:, i, :, :] > upperbound[i], upperbound[i], datapctlimit[:, i, :, :])
        datapctlimit[:, i, :, :] = np.where(datapctlimit[:, i, :, :] < lowerbound[i], lowerbound[i], datapctlimit[:, i, :, :])
    for i in range(len(upperbound)):
        print("post-truncation min:",datapctlimit[:, i, :, :].min())
        print("post-truncation max:", datapctlimit[:, i, :, :].max())
    # datanorm2 = cv2.normalize(datapctlimit, datapctlimit, -1, 1, cv2.NORM_MINMAX)
    return datapctlimit

def ecg_truncate_mean_normalize_batch(paths, output_dir='./processed_data/ekg_bwr_trunc_norm'):
    os.makedirs(output_dir, exist_ok=True)
    dataset_data = np.stack([np.load(path) for path in paths])
    if dataset_data.shape[1:] == (2500,12,1):
        dataset_data = np.transpose(dataset_data, axes=[0,2,1,3])
    elif dataset_data.shape[1:] == (1,2500,12):
        dataset_data = np.transpose(dataset_data, axes=[0,3,2,1])
    elif dataset_data.shape[1:] == (12,2500,1):
        pass
    assert dataset_data.shape[1:] == (12, 2500, 1), "dataset is not X,12,2500,1"

    # Compute mean and standard deviation for normalization
    mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD before truncation')
    print(mean_tr)
    print(std_tr)

    lowerbound = []
    upperbound = []
    for i in range(0,12):
        dataset = dataset_data
        lowerbound.append(np.percentile(dataset[:,i,:,:],0.1))
        upperbound.append(np.percentile(dataset[:,i,:,:],99.9))

    print("dataset 0.1st percentile lowerbound:",lowerbound)
    print("dataset 99.9th percentile upperbound:",upperbound)


    dataset_data = truncate_data_percentile(dataset_data, lowerbound, upperbound)
    print("dataset data truncated based on percentiles")

    mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD before normalization', mean_tr, std_tr)

    # Per-lead normalization (there are 12 leads)
    assert (len(mean_tr) == dataset_data.shape[1])
    for i in range(len(mean_tr)):
        tic = time.perf_counter()
        print("pre-normalizing min:",dataset_data[:, i, :, :].min())
        print("pre-normalizing max:", dataset_data[:, i, :, :].max())
        dataset_data[:, i, :, :] = (dataset_data[:, i, :, :] - mean_tr[i]) / std_tr[i]
        print("post-normalizing min:",dataset_data[:, i, :, :].min())
        print("post-normalizing max:", dataset_data[:, i, :, :].max())
        toc = time.perf_counter()
        print(f'Processed iter {i} in {toc - tic:0.2f} seconds')

    mean_norm, std_norm = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD after normalization', mean_norm, std_norm)
    
    # Transpose data for model
    # Not sure if we need this still 
    X_dataset = np.transpose(dataset_data, axes=[0, 3, 2, 1])
    trunc_norm_paths = []
    for i in range(len(dataset_data)):
        original_path = paths[i].split('/')[-1]
        trunc_norm_path = output_dir + '/' + original_path
        trunc_norm_paths.append(trunc_norm_path)
        np.save(trunc_norm_path, dataset_data[i])
    return trunc_norm_paths

BWR_TRUNC_NORM_PATH = './processed_data/ekg_bwr_trunc_norm'
os.makedirs(BWR_TRUNC_NORM_PATH, exist_ok=True)


def ecg_truncate_mean_normalize_batch(paths, output_dir='./processed_data/ekg_bwr_trunc_norm'):
    os.makedirs(output_dir, exist_ok=True)
    dataset_data = np.stack([np.load(path) for path in paths])
    if dataset_data.shape[1:] == (2500,12,1):
        dataset_data = np.transpose(dataset_data, axes=[0,2,1,3])
    elif dataset_data.shape[1:] == (1,2500,12):
        dataset_data = np.transpose(dataset_data, axes=[0,3,2,1])
    elif dataset_data.shape[1:] == (12,2500,1):
        pass
    assert dataset_data.shape[1:] == (12, 2500, 1), "dataset is not X,12,2500,1"

    # Compute mean and standard deviation for normalization
    mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD before truncation')
    print(mean_tr)
    print(std_tr)

    lowerbound = []
    upperbound = []
    for i in range(0,12):
        dataset = dataset_data
        lowerbound.append(np.percentile(dataset[:,i,:,:],0.1))
        upperbound.append(np.percentile(dataset[:,i,:,:],99.9))

    print("dataset 0.1st percentile lowerbound:",lowerbound)
    print("dataset 99.9th percentile upperbound:",upperbound)


    dataset_data = truncate_data_percentile(dataset_data, lowerbound, upperbound)
    print("dataset data truncated based on percentiles")

    mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD before normalization', mean_tr, std_tr)

    # Per-lead normalization (there are 12 leads)
    assert (len(mean_tr) == dataset_data.shape[1])
    for i in range(len(mean_tr)):
        tic = time.perf_counter()
        print("pre-normalizing min:",dataset_data[:, i, :, :].min())
        print("pre-normalizing max:", dataset_data[:, i, :, :].max())
        dataset_data[:, i, :, :] = (dataset_data[:, i, :, :] - mean_tr[i]) / std_tr[i]
        print("post-normalizing min:",dataset_data[:, i, :, :].min())
        print("post-normalizing max:", dataset_data[:, i, :, :].max())
        toc = time.perf_counter()
        print(f'Processed iter {i} in {toc - tic:0.2f} seconds')

    mean_norm, std_norm = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD after normalization', mean_norm, std_norm)
    
    # Transpose data for model
    # Not sure if we need this still 
    X_dataset = np.transpose(dataset_data, axes=[0, 3, 2, 1])
    trunc_norm_paths = []
    for i in range(len(dataset_data)):
        original_path = paths[i].split('/')[-1]
        trunc_norm_path = output_dir + '/' + original_path
        trunc_norm_paths.append(trunc_norm_path)
        np.save(trunc_norm_path, dataset_data[i])
    return trunc_norm_paths


import time
def ecg_truncate_mean_normalize_path(path):
    dataset_data = np.expand_dims(np.load(path), 0)
    dataset_data = np.expand_dims(dataset_data, 3)
    print(dataset_data.shape)
    if dataset_data.shape[1:] == (2500,12,1):
        dataset_data = np.transpose(dataset_data, axes=[0,2,1,3])
    elif dataset_data.shape[1:] == (1,2500,12):
        dataset_data = np.transpose(dataset_data, axes=[0,3,2,1])
    elif dataset_data.shape[1:] == (12,2500,1):
        pass
    assert dataset_data.shape[1:] == (12, 2500, 1), "dataset is not X,12,2500,1"

    # Compute mean and standard deviation for normalization
    mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD before truncation')
    print(mean_tr)
    print(std_tr)

    lowerbound = []
    upperbound = []
    for i in range(0,12):
        dataset = dataset_data
        lowerbound.append(np.percentile(dataset[:,i,:,:],0.1))
        upperbound.append(np.percentile(dataset[:,i,:,:],99.9))

    print("dataset 0.1st percentile lowerbound:",lowerbound)
    print("dataset 99.9th percentile upperbound:",upperbound)


    dataset_data = truncate_data_percentile(dataset_data, lowerbound, upperbound)
    # print("dataset data truncated based on percentiles")

    # mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    # # print('Mean and STD before normalization', mean_tr, std_tr)

    mean_tr = np.array([ 4.94534396,  4.43621112, -0.38007734, -4.6947857 ,  2.62600571,
        1.97369681, -3.98433206, -1.0012726 ,  0.24998725,  3.24802075,
        4.91321112,  5.29477241])
    

    # Taken from PreOpNet
    mean_tr = np.array([-0.753910531911661,-0.5609376530271284,0.19297287888453685,
                        0.6574240924693946,-0.09648643944226842,0.09648643944226842,
                        -0.9398182103547104,-0.8866948773251518,-0.9585726095365399,
                        -0.9142084935751398,-0.9573448180888456,-0.9300810636208064])
   

    std_tr = np.array([28.23379939, 32.03115308, 31.30188996, 25.86865272, 25.15370549,
        28.36177382, 37.31252219, 53.95075544, 51.42862679, 43.52826587,
        38.84172535, 34.50052327])
    
    # Taken from PreOpNet
    std_tr = np.array([32.082092358503644,34.97862852596865,38.153409045189754,
                       27.612712586528637,19.076704522594877,19.076704522594877,
                       42.77931881050877,63.872623440588406,61.15731462396783,
                       54.12879139607189,48.435274440820855,43.34056377213695])
    
    print(mean_tr.shape)
    print(dataset_data.shape[1])

    # Per-lead normalization (there are 12 leads)
    assert (len(mean_tr) == dataset_data.shape[1])
    for i in range(len(mean_tr)):
        # tic = time.perf_counter()
        # print("pre-normalizing min:",dataset_data[:, i, :, :].min())
        # print("pre-normalizing max:", dataset_data[:, i, :, :].max())
        dataset_data[:, i, :, :] = (dataset_data[:, i, :, :] - mean_tr[i]) / std_tr[i]
        # print("post-normalizing min:",dataset_data[:, i, :, :].min())
        # print("post-normalizing max:", dataset_data[:, i, :, :].max())
        # toc = time.perf_counter()
        # print(f'Processed iter {i} in {toc - tic:0.2f} seconds')

    mean_norm, std_norm = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD after normalization', mean_norm, std_norm)
    
    # Transpose data for model
    # Not sure if we need this still 
    original_path = path.split('/')[-1]
    trunc_norm_path = BWR_TRUNC_NORM_PATH + '/' + original_path
    np.save(trunc_norm_path, dataset_data[0,:,:,0])
    return trunc_norm_path

def calc_mean_tr_std_tr(paths):
    dataset_data = np.stack([np.load(path) for path in paths])
    dataset_data = np.expand_dims(dataset_data, 3)
    print(dataset_data.shape)
    if dataset_data.shape[1:] == (2500,12,1):
        dataset_data = np.transpose(dataset_data, axes=[0,2,1,3])
    elif dataset_data.shape[1:] == (1,2500,12):
        dataset_data = np.transpose(dataset_data, axes=[0,3,2,1])
    elif dataset_data.shape[1:] == (12,2500,1):
        pass
    assert dataset_data.shape[1:] == (12, 2500, 1), "dataset is not X,12,2500,1"

    # Compute mean and standard deviation for normalization
    mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD before truncation')

    lowerbound = []
    upperbound = []
    for i in range(0,12):
        dataset = dataset_data
        lowerbound.append(np.percentile(dataset[:,i,:,:],0.1))
        upperbound.append(np.percentile(dataset[:,i,:,:],99.9))

    print("dataset 0.1st percentile lowerbound:",lowerbound)
    print("dataset 99.9th percentile upperbound:",upperbound)


    dataset_data = truncate_data_percentile(dataset_data, lowerbound, upperbound)
    print("dataset data truncated based on percentiles")

    mean_tr, std_tr = np.mean(dataset_data, axis=(0, 2, 3)), np.std(dataset_data, axis=(0, 2, 3))
    print('Mean and STD before normalization', mean_tr, std_tr)
    return mean_tr, std_tr

def create_name_dob_hash(df, first_name_col, last_name_col, dob_col):
    df[first_name_col] = df[first_name_col].str.lower()
    df[last_name_col] = df[last_name_col].str.lower()
    df[dob_col] = pd.to_datetime(df[dob_col])
    df[dob_col] = df[dob_col].dt.strftime('%m-%d-%Y')
    df['name_dob_hash'] = df[first_name_col] + '_' + df[last_name_col] + '_' + df[dob_col]
    df['name_dob_hash'] = df['name_dob_hash'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
    return df['name_dob_hash']