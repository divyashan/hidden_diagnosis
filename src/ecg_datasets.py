from torch.utils.data import Dataset
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, ecg_paths, labels, root_dir='./processed_data/ekg_bwr_trunc_norm', transform=None):
        self.ecg_paths = ecg_paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ecg_paths)

    def __getitem__(self, idx): 
        ecg_path = self.ecg_paths[idx]
        ecg = np.load(ecg_path, allow_pickle=True)
        label = self.labels[idx]

        # Half the time, introduce transform
        if self.transform and np.random.random() > .5:
            ecg = self.transform(ecg) 

        return ecg, label
    
class ECGOneLeadDataset(Dataset):
    def __init__(self, ecg_paths, labels, root_dir='./processed_data/ekg_bwr_trunc_norm', transform=None):
        self.ecg_paths = ecg_paths
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ecg_paths)

    def __getitem__(self, idx): 
        ecg_path = self.ecg_paths[idx]
        ecg = np.load(ecg_path)
        label = self.labels[idx]


        return ecg[0:1], label
    
class ECGDemographicsDataset(Dataset):
    def __init__(self, ecg_paths, extra_feats, labels, root_dir='./processed_data/ekg_bwr_trunc_norm'):
        self.ecg_paths = ecg_paths
        self.additional_feats = extra_feats
        self.labels = labels
        self.root_dir = root_dir

    def __len__(self):
        return len(self.ecg_paths)

    def __getitem__(self, idx): 
        ecg_path = self.ecg_paths[idx]
        ecg = np.load(ecg_path, allow_pickle=True)
        # assert ecg.shape == (12, 2500)
        # if ecg.shape != (12, 2500):
        #     print("BIG PROBLEM ecg shape is ", ecg.shape)
        additional_feats = self.additional_feats[idx]
        label = self.labels[idx]
        return (ecg, additional_feats), label
