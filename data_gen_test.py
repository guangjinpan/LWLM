import sys
sys.path.append('../src/')

import DeepMIMOv3 as DeepMIMO
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import h5py
import music

# Load the default parameters
parameters = DeepMIMO.default_params()

# Set scenario name
parameters['scenario'] = 'O1_3p5'

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'../scenarios'


parameters['num_paths'] = 5

# User rows 1-100
parameters['user_rows'] = np.arange(2700)
parameters['user_subsampling'] = 0.01 # Subsample users to 1% of the original number


# Activate only the first basestation
BS_list=[3]
print("BS_list=",BS_list)
parameters['active_BS'] = np.array(BS_list) 

BW = 10
parameters['OFDM']['bandwidth'] = BW/1000 # 100 MHz
parameters['OFDM']['subcarriers'] = 128 # OFDM with 128 subcarriers
parameters['OFDM']['selected_subcarriers'] = np.arange(128) # Keep only first 64 subcarriers

N_antenna=32
parameters['ue_antenna']['shape'] = np.array([1, 1]) # Single antenna
parameters['bs_antenna']['shape'] = np.array([N_antenna, 1]) # ULA of 32 elements
parameters['enable_BS2B'] = 0
pprint(parameters)

# Generate data
dataset = DeepMIMO.generate_data(parameters)




for BS_i in range(len(dataset)):
    bs_i = BS_list[BS_i]
    for ue_i in range(len(dataset[BS_i]['user']['channel'])):
        # print(ue_i)
        file_name= "../../LWLM/dataset/O1_3p5/test"
        with h5py.File(file_name+f"/BS{bs_i}_{parameters['OFDM']['subcarriers']}sc_{N_antenna}at_{BW}M/{bs_i}_{ue_i}.h5py", 'w') as f:
            f.create_dataset('channel_real', data=np.real(dataset[BS_i]['user']['channel'][ue_i]))  # 实部
            f.create_dataset('channel_imag', data=np.imag(dataset[BS_i]['user']['channel'][ue_i]))  # 虚部
            f.create_dataset('UElocation', data=dataset[BS_i]['user']['location'][ue_i])  # 
            f.create_dataset('LoS', data=dataset[BS_i]['user']['LoS'][ue_i])  # 
            f.create_dataset('distance', data=dataset[BS_i]['user']['distance'][ue_i])  # 
            f.create_dataset('BSlocation', data=dataset[BS_i]['location'])
            f.create_dataset('DoD_phi', data=dataset[BS_i]['user']["paths"][ue_i]["DoD_phi"])
            f.create_dataset('DoD_theta', data=dataset[BS_i]['user']["paths"][ue_i]["DoD_theta"])
            f.create_dataset('bandwidth', data=BW)




