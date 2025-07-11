import hashlib 

SAVED_MODELS_DIR = '/home/dy516/Documents/care_cascade/saved_models'
PREPROCESSED_DATA_DIR = './processed_data/'
RESULTS_DIR = './results/'

def map_params_to_filename(params):
    sorted_keys = sorted(params.keys())
    return '_'.join([str(k) + ',' + str(params[k]) for k in sorted_keys])

def map_params_to_filename_hash(params):
    sorted_keys_str = map_params_to_filename(params)
    sha256 = hashlib.sha256()
    sha256.update(sorted_keys_str.encode('utf-8'))
    hashed_string = sha256.hexdigest()
    return hashed_string

