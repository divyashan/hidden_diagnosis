
import os
from bs4 import BeautifulSoup
import xmltodict
import struct
import numpy as np
import base64
import pdb
from typing import List, Dict

home_dir = os.getcwd()
npy_output_directory_path = os.path.join(home_dir,'ekg_waveform_output') #this can be left as default

dir_name = npy_output_directory_path+'/'

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


ecg_feature_names = ['VentricularRate',
                   'AtrialRate',
                   'PRInterval',
                   'QRSDuration',
                   'QTInterval',
                   'QTCorrected',
                   'PAxis',
                   'RAxis',
                   'TAxis',
                   'QRSCount',
                   'QOnset',
                   'QOffset',
                   'POnset',
                   'POffset',
                   'TOffset',
                   'ECGSampleBase']

demographic_feature_names = ['Race',
                            'Gender',
                            'Age']

demographic_binarized_feature_names = ['binary_Race_CAUCASIAN',
                                    'binary_Race_BLACK',
                                    'binary_Race_HISPANIC',
                                    'binary_Gender_MALE',
                                    'Age']

troponin_feature_names = ['trop_abnormal', 'trop_test']
ecg_metadata_names = ['DataType',
                      'LocationName']

def map_params_to_filename(params):
    sorted_keys = sorted(params.keys())
    return '_'.join([str(k) + ',' + str(params[k]) for k in sorted_keys])

def xml_file_to_name_dob(filepath):
    bs = BeautifulSoup(open(filepath, 'r').read(), 'xml')
    feature_dict = {}
    name_dob_feature_names = ['PatientFirstName', 'PatientLastName', 'DateofBirth', 'PatientID']
    for feat_name in name_dob_feature_names:
        try:
            feature_dict[feat_name] = bs.find(feat_name).text
        except:
            feature_dict[feat_name] = None 
    return feature_dict

def xml_file_to_features(filepath):
    # lead_data_path = xml_to_np_array_file(filepath, path_to_output='./processed_data/ekg_waveform_output')
    lead_data_path = xml_to_np_array_file(filepath, path_to_output='/data/workspace/ekg_waveform_output')
    bs = BeautifulSoup(open(filepath, 'r').read(), 'xml')
    feature_dict = {}
    for feat_name in ecg_feature_names:
        try:
            feature_dict[feat_name] = int(bs.find(feat_name).text)
        except:
            feature_dict[feat_name] = None 
    for feat_name in demographic_feature_names:
        try:
            feature_dict[feat_name] = bs.find(feat_name).text
        except:
            feature_dict[feat_name] = None
    for feat_name in ecg_metadata_names:
        try:
            feature_dict[feat_name] = bs.find(feat_name).text
        except:
            feature_dict[feat_name] = None 
    
    data_types = bs.find_all('DataType')
    data_types = ','.join([tag.text for tag in data_types]).lower()
    feature_dict['resting_flag'] = 'resting' in data_types
    
    diagnosis_tags = bs.find_all('DiagnosisStatement')
    diagnosis_statements = ','.join([tag.text for tag in diagnosis_tags]).lower()
    # sinus_flag = 'sinus rhythm' in diagnosis_statements
    sinus_flag = np.any([phrase in diagnosis_statements for phrase in SINUS_RHYTHM_PHRASES]) 
    sinus_flag_kws = [phrase for phrase in SINUS_RHYTHM_PHRASES if phrase in diagnosis_statements]
    feature_dict['sinus_rhythm_flag'] = sinus_flag
    feature_dict['sinus_rhythm_flag_kwws'] = sinus_flag_kws

    return feature_dict, lead_data_path

def xml_file_to_extracted_features(filepath):
    bs = BeautifulSoup(open(filepath, 'r').read(), 'xml')
    feature_dict = {}
    for feat_name in ecg_feature_names:
        try:
            feature_dict[feat_name] = int(bs.find(feat_name).text)
        except:
            feature_dict[feat_name] = None 
    for feat_name in demographic_feature_names:
        try:
            feature_dict[feat_name] = bs.find(feat_name).text
        except:
            feature_dict[feat_name] = None
    for feat_name in ecg_metadata_names:
        try:
            feature_dict[feat_name] = bs.find(feat_name).text
        except:
            feature_dict[feat_name] = None 
    
    data_types = bs.find_all('DataType')
    data_types = ','.join([tag.text for tag in data_types]).lower()
    feature_dict['resting_flag'] = 'resting' in data_types
    
    diagnosis_tags = bs.find_all('DiagnosisStatement')
    diagnosis_statements = ','.join([tag.text for tag in diagnosis_tags]).lower()
    # sinus_flag = 'sinus rhythm' in diagnosis_statements
    sinus_flag = np.any([phrase in diagnosis_statements for phrase in SINUS_RHYTHM_PHRASES]) 
    sinus_flag_kws = [phrase for phrase in SINUS_RHYTHM_PHRASES if phrase in diagnosis_statements]
    feature_dict['sinus_rhythm_flag'] = sinus_flag
    feature_dict['sinus_rhythm_flag_kwws'] = sinus_flag_kws

    return feature_dict


AFIB_PHRASES = ['atrial fibrillation with rapid ventricular response', 'atrial fibrillation with moderate ventricular response',  
                    'fibrillation/flutter',  'atrial fibrillation with controlled ventricular response', 
                    'afib', 'atrial fib', 'afibrillation', 'atrial fibrillation', 'atrialfibrillation']

SINUS_RHYTHM_PHRASES = ["conducted sinus impulses", "marked sinus arrhythmia", 
                        "normal when compared with ecg of", "rhythm remains normal sinus", 
                        "frequent native sinus beats", "normal ecg", "atrialbigeminy", 
                        "sa block", "atrial trigeminy", "sinus tachycardia", "sinus rhythm", "sinus bradycardia",
                        "rhythm has reverted to normal", "rhythm is now clearly sinus", "sinus exit block", 
                        "tracing is within normal limits", "1st degree sa block", "sinus arrhythmia", 
                        "2nd degree sa block", 
                        "tracing within normal limits", "sinus mechanism has replaced", "atrial bigeminal rhythm", 
                        "sa exit block", "sinoatrial block", "rhythm is normal sinus", "with occasional native sinus beats", 
                        "sinus slowing", "atrial bigeminy and ventricular bigeminy"]

# https://github.com/mit-ccrg/ml4c3-mirror/blob/ea23e26eeb9814bb057f2c86c62c923dd347aa95/tensormap/ecg_labels.py#L62

STEMI_PHRASES = ['STEMI', 'ST elevation acute MI']

def map_xml_path_to_afib(xml_path):
    diagnosis_statements = xml_file_to_all_diagnosis_statements(xml_path).lower()
    afib_flag = np.any([phrase in diagnosis_statements for phrase in AFIB_PHRASES]) 
    return int(afib_flag)

def map_xml_path_to_afib_with_rvr(xml_path):
    diagnosis_statements = xml_file_to_all_diagnosis_statements(xml_path).lower()
    afib_flag = np.any([phrase in diagnosis_statements for phrase in AFIB_PHRASES]) 
    rvr_flag = 'rapid ventricular response' in diagnosis_statements
    return int(afib_flag and rvr_flag)

def map_xml_path_to_stemi(xml_path):
    diagnosis_statements = xml_file_to_all_diagnosis_statements(xml_path).lower()
    stemi_flag = np.any([phrase in diagnosis_statements for phrase in STEMI_PHRASES]) 
    return int(stemi_flag)

def xml_file_to_all_diagnosis_statements(filepath):
    bs = BeautifulSoup(open(filepath, 'r').read(), 'xml')
    diagnosis_tags = bs.find_all('DiagnosisStatement')
    diagnosis_statements = ','.join([tag.text for tag in diagnosis_tags])
    return diagnosis_statements 

def add_binary_demographic_features(sample):
    for binarized_feature_name in demographic_binarized_feature_names:
        if 'binary' in binarized_feature_name:
            feature_name = binarized_feature_name.split('_')[1]
            feature_value = binarized_feature_name.split('_')[2]
            sample[binarized_feature_name] = sample[feature_name] == feature_value
        else:
            sample[binarized_feature_name] = sample[binarized_feature_name]
    return sample



# From IntroECG repo
def decode_ekg_muse(raw_wave):
    """
    Ingest the base64 encoded waveforms and transform to numeric
    """
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char*int(len(arr)/2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols,  arr)
    return byte_array

# From IntroECG repo
def decode_ekg_muse_to_array(raw_wave, downsample = 1):
    """
    Ingest the base64 encoded waveforms and transform to numeric

    downsample: 0.5 takes every other value in the array. Muse samples at 500/s and the sample model requires 250/s. So take every other.
    """
    try:
        dwnsmpl = int(1//downsample)
    except ZeroDivisionError:
        print("You must downsample by more than 0")
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, 'utf-8'))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = ''.join([char*int(len(arr)/2) for char in 'h'])
    byte_array = struct.unpack(unpack_symbols,  arr)
    return np.array(byte_array)[::dwnsmpl]


# From IntroECG repo
def xml_to_np_array_file(path_to_xml, path_to_output = home_dir):

    with open(path_to_xml, 'rb') as fd:
        dic = xmltodict.parse(fd.read().decode('utf8'))

    """

    Upload the ECG as numpy array with shape=[2500,12,1] ([time, leads, 1]).

    The voltage unit should be in 1 mv/unit and the sampling rate should be 250/second (total 10 second).

    The leads should be ordered as follow I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.

    """
    try:
        pt_id = dic['RestingECG']['PatientDemographics']['PatientID']
    except:
        print("no PatientID")
        pt_id = "none"
    try:
        PharmaUniqueECGID = dic['RestingECG']['PharmaData']['PharmaUniqueECGID']
    except:
        print("no PharmaUniqueECGID")
        PharmaUniqueECGID = "none"
    try:
        AcquisitionDateTime = dic['RestingECG']['TestDemographics']['AcquisitionDate'] + "_" + dic['RestingECG']['TestDemographics']['AcquisitionTime'].replace(":","-")
    except:
        print("no AcquisitionDateTime")
        AcquisitionDateTime = "none"    
        
    #need to instantiate leads in the proper order for the model
    lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    filename = '{}_{}_{}.npy'.format(pt_id, AcquisitionDateTime,PharmaUniqueECGID)
    path_to_output += '/'+filename
    if os.path.exists(path_to_output +'.npy'):
        return path_to_output
    """
    Each EKG will have this data structure:
    lead_data = {
        'I': np.array
    }
    """
    try:
        if 'Waveform' not in dic['RestingECG'].keys():
            return
        lead_data =  dict.fromkeys(lead_order)
        for lead in dic['RestingECG']['Waveform']:
            for leadid in range(len(lead['LeadData'])):
                    sample_length = len(decode_ekg_muse_to_array(lead['LeadData'][leadid]['WaveFormData']))
                    #sample_length is equivalent to dic['RestingECG']['Waveform']['LeadData']['LeadSampleCountTotal']
                    if sample_length == 5000:
                        lead_data[lead['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(lead['LeadData'][leadid]['WaveFormData'], downsample = 0.5)
                    elif sample_length == 2500:
                        lead_data[lead['LeadData'][leadid]['LeadID']] = decode_ekg_muse_to_array(lead['LeadData'][leadid]['WaveFormData'], downsample = 1)
                    else:
                        continue
                #ensures all leads have 2500 samples and also passes over the 3 second waveform
    except:
        return
    lead_data['III'] = (np.array(lead_data["II"]) - np.array(lead_data["I"]))
    lead_data['aVR'] = -(np.array(lead_data["I"]) + np.array(lead_data["II"]))/2
    lead_data['aVF'] = (np.array(lead_data["II"]) + np.array(lead_data["III"]))/2
    lead_data['aVL'] = (np.array(lead_data["I"]) - np.array(lead_data["III"]))/2

    lead_data = {k: lead_data[k] for k in lead_order}
    # drops V3R, V4R, and V7 if it was a 15-lead ECG

    # now construct and reshape the array
    # converting the dictionary to an npy.array
    temp = []
    for key,value in lead_data.items():
        temp.append(value)

    #transpose to be [time, leads, ]
    ekg_array = np.array(temp).T

    #expand dims to [time, leads, 1]
    ekg_array = np.expand_dims(ekg_array, axis=-1)

    # Here is a check to make sure all the model inputs are the right shape
#     assert ekg_array.shape == (2500, 12, 1), "ekg_array is shape {} not (2500, 12, 1)".format(ekg_array.shape )

    if ekg_array.shape != (2500, 12, 1):
        return None
    filename = '{}_{}_{}.npy'.format(pt_id, AcquisitionDateTime,PharmaUniqueECGID)
    if not os.path.exists(path_to_output):
        with open(path_to_output, 'wb') as f:
            np.save(f, ekg_array)
    return path_to_output

### Self-supervised representations 

from tensorflow.keras.models import load_model, Model
import os
LEADS = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
]
MODEL = load_model("./ml4h/model_zoo/PCLR/PCLR.h5", compile=False)

def process_ecg(ecg: Dict[str, np.ndarray], ecg_samples: int = 4096) -> np.ndarray:
    """
    Prepares an ECG for use in a tensorflow model
    :param ecg: A dictionary mapping lead name to lead values.
                The lead values should be measured in milli-volts.
                Each lead should represent 10s of samples.
    :param ecg_samples: Length of each lead for input into the model.
    :return: a numpy array of the ECG shaped (ecg_samples, 12)
    """
    assert set(ecg.keys()) == set(LEADS)

    out = np.zeros((ecg_samples, 12))
    for i, lead_name in enumerate(LEADS):
        lead = ecg[lead_name]
        interpolated_lead = np.interp(
            np.linspace(0, 1, ecg_samples),
            np.linspace(0, 1, lead.shape[0]),
            lead,
        )
        out[:, i] = interpolated_lead
    return out

def get_model() -> Model:
    """Get PCLR embedding model"""
    return load_model("./ml4h/model_zoo/PCLR/PCLR.h5", compile=False)


def get_representations(ecgs: List[Dict[str, np.ndarray]]) -> np.ndarray:
    """
    Uses PCLR trained model to build representations of ECGs
    :param ecgs: A list of dictionaries mapping lead name to lead values.
                 The lead values should be measured in milli-volts.
                 Each lead should represent 10s of samples.
    :return:
    """
    model = MODEL
    ecgs = np.stack(list(map(process_ecg, ecgs)))
    return model.predict(ecgs)


def ecg_arr_to_ecg_dict(ecg_arr):
    # Input: ecg_arr,  2500 x 12 x 1 array of lead values
    # Output: ecg_dict, dictionary mapping lead name to lead array
    ecg_dict = {lead_name : ecg_arr[:,i,0] for i, lead_name in enumerate(LEAD_ORDER)}
    return ecg_dict

def get_list_of_ecg_arrs(df):
    list_of_ecg_paths = list(df['path_to_lead_data'])
    list_of_ecg_arrs = [np.load(x) for x in list_of_ecg_paths]
    return list_of_ecg_arrs
