import numpy as np
import torch
import os
import math
import torch
# from tqdm import tqdm
from torch.utils.data import ConcatDataset
#import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.pyplot as plt
from train_model import Wrapper

 
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
sys.path.append('../pre_data/')
from data_path import datapath
import h5py



def channel_normalization(H):
    P_current = np.sum(np.abs(H) ** 2)
    # alpha = np.sqrt((H.shape[1] * H.shape[2]) / P_current)
    alpha = np.sqrt((H.shape[2]) / P_current)

    # Scale the matrix so that the total power becomes N*M
    H_normalized = H * alpha
    return H_normalized  


def channel_normalization_with_noise(H, snr_db):
    P_current = np.sum(np.abs(H) ** 2)
    # alpha = np.sqrt((H.shape[1] * H.shape[2]) / P_current)
    # alpha = np.sqrt((H.shape[2]) / P_current)
    alpha = np.sqrt((H.shape[1] * H.shape[2] / 2) / P_current)

    # Scale the matrix so that the total power becomes N*M
    H_normalized = H * alpha

    # Signal power (normalized to approximately H.shape[2])
    signal_power = np.mean(np.abs(H_normalized) ** 2)
    
    # Compute noise power
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian white noise (real and imaginary parts are independent, mean 0, variance noise_power/2)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*H.shape) + 1j * np.random.randn(*H.shape))
    
    # Add noise
    H_noisy = H_normalized + noise 

    return H_noisy  

def freq_to_delay_angle_domain(H_freq):
    H_complex = H_freq[0] + 1j * H_freq[1]

    H_delay = np.fft.ifft(H_complex, axis=0)

    H_delay_angle_complex = np.fft.fft(H_delay, axis=1) / np.sqrt(H_delay.shape[0]) #/ 2

    return H_delay_angle_complex


class generate_Dataset(Dataset):
    def __init__(self, EnvPara, is_val =0):
        
        self.input_fmap = EnvPara["input_fmap"]
        self.input_tdim = EnvPara["input_tdim"]
        self.input_fdim = EnvPara["input_fdim"]
        self.task = EnvPara["task"]
        self.EnvPara = EnvPara
        self.is_val = is_val
        if self.is_val:
            self.data_list = np.load("../pre_data/val_1000.npy")
            print("../pre_data/val_1000.npy")
        elif self.task == "SingleBSLoc":
            self.data_list = np.load("../pre_data/train.npy")
            self.data_list = self.data_list[ : EnvPara["FT_dataset"]]
        elif self.task == "Compression":
            self.data_list = np.load("../pre_data/train.npy")
            self.data_list = self.data_list[ : EnvPara["FT_dataset"]]
        elif self.task == "MultiBSLoc":
            self.data_list = np.load("../pre_data/train.npy")
            self.data_list = self.data_list[ : EnvPara["FT_dataset"]]
        elif self.task == "cnn":
            self.data_list = np.load("../pre_data/train.npy")
            self.data_list = self.data_list[ : EnvPara["FT_dataset"]]
        elif self.task == "toa":
            self.data_list = np.load("../pre_data/train.npy")
            self.data_list = self.data_list[ : EnvPara["FT_dataset"]]
        elif self.task == "aoa":
            self.data_list = np.load("../pre_data/train.npy")
            self.data_list = self.data_list[ : EnvPara["FT_dataset"]]
        else:
            self.data_list = np.load("../pre_data/pretrain_data.npy")

        self.len = len(self.data_list)
        print("data_list",len(self.data_list))
        self.BS_list = [3, 4, 9, 10]
        self.BW_list = [10, 20, 50]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        

        ### fine-tune data
        if (self.task == "SingleBSLoc") | (self.task == "aoa") | (self.task == "toa") | (self.task == "MultiBSLoc")|(self.task == "cnn") : 
            
            random_number = self.data_list[idx]
            random_BW_1 = self.EnvPara["BW"]
            data_path = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/DeepMIMO/O1_3p5/"
            channel_data_aa = np.zeros((self.EnvPara["BS_Num"],self.input_fmap, self.input_tdim, self.input_fdim))
            channel_Antenna_subcarrier_aa = np.zeros((self.EnvPara["BS_Num"], self.input_fmap, self.input_tdim, self.input_fdim))
            channel_Antenna_subcarrier_ri = np.zeros((self.EnvPara["BS_Num"], self.input_fmap, self.input_tdim, self.input_fdim))
            channel_delay_angle_ri = np.zeros((self.EnvPara["BS_Num"], self.input_fmap, self.input_tdim, self.input_fdim))        
            UElocation_all = np.zeros([self.EnvPara["BS_Num"],4])
            BSconf_all = np.zeros([self.EnvPara["BS_Num"],3])

            # load data
            for BS_i in range(self.EnvPara["BS_Num"]):
                random_BS_1 = self.BS_list[BS_i]
                with h5py.File(data_path+f"BS{random_BS_1}_128sc_32at_{random_BW_1}M"+f"/{random_BS_1}_{random_number}.h5py", 'r') as f:
                    channel_real = f["channel_real"][:]
                    channel_imag = f["channel_imag"][:]
                    channel_1 = channel_real+ 1j* channel_imag
                    UElocation_1 = f["UElocation"][:][:2]
                    BSlocation_1 = f["BSlocation"][:][:2]
                    distance_1 = f["distance"][()]
                    angle_1 = f['DoD_phi'][()][0]


                # channel normalization
                channel_1 = channel_normalization(channel_1)


                # CFR with amplitude and phase 
                channel_aa_1 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
                channel_aa_1[0,:,:] = np.abs(channel_1).astype(np.float32)
                channel_aa_1[1,:,:] = np.angle(channel_1).astype(np.float32)

                # CFR with real and imaginary
                channel_ri_1 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
                channel_ri_1[0,:,:] = np.real(channel_1).astype(np.float32)
                channel_ri_1[1,:,:] = np.imag(channel_1).astype(np.float32)   

                # delay angle domain data with real and imaginary
                channel_delay_angle_1 = freq_to_delay_angle_domain(channel_ri_1)
                channel_delay_angle_ri_1 = np.zeros([self.input_fmap, channel_ri_1.shape[1], channel_ri_1.shape[2]])
                channel_delay_angle_ri_1[0,:,:] = np.real(channel_delay_angle_1).astype(np.float32)
                channel_delay_angle_ri_1[1,:,:] = np.imag(channel_delay_angle_1).astype(np.float32)

                # spatial-frequency domain data with amplitude and phase 
                channel_Antenna_subcarrier_aa[BS_i,:] = channel_aa_1.copy()
                # spatial-frequency domain data with real and imaginary
                channel_Antenna_subcarrier_ri[BS_i,:] = channel_ri_1.copy()
                # delay angle domain data with real and imaginary
                channel_delay_angle_ri[BS_i,:] = channel_delay_angle_ri_1.copy()

                # labels: 0,1:BSlocation, 2:Bandwidth
                UElocation_all[BS_i,:2] = UElocation_1.copy()/10   
                UElocation_all[BS_i, 2] = distance_1 /100   
                UElocation_all[BS_i, 3] = angle_1 /100
                # configurations 0,1:BSlocation, 2:Bandwidth
                BSconf_all[BS_i,:2] = BSlocation_1.copy() / 100  
                BSconf_all[BS_i, 2] = random_BW_1 / 10




        ### pretrain data             
        else:
            while (1):
                # spatial-frequency domain data with amplitude and phase
                channel_Antenna_subcarrier_aa = np.zeros((2, self.input_fmap, self.input_tdim, self.input_fdim))
                channel_Antenna_subcarrier_ri = np.zeros((2, self.input_fmap, self.input_tdim, self.input_fdim))
                channel_delay_angle_ri = np.zeros((2, self.input_fmap, self.input_tdim, self.input_fdim))  


                # load data
                random_number = self.data_list[random.randint(0, len(self.data_list)-1)]
                random_BS_1 = self.BS_list[random.randint(0, len(self.BS_list)-1)]
                random_BW_1 = self.BW_list[random.randint(0, len(self.BW_list)-1)]
                data_path = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/DeepMIMO/O1_3p5/"
                with h5py.File(data_path+f"BS{random_BS_1}_128sc_32at_{random_BW_1}M"+f"/{random_BS_1}_{random_number}.h5py", 'r', swmr=True) as f:
                    channel_real = f["channel_real"][:]
                    channel_imag = f["channel_imag"][:]
                    channel_1 = channel_real+ 1j* channel_imag
                    UElocation_1 = f["UElocation"][:][:2]
                    BSlocation_1 = f["BSlocation"][:][:2] 
                    distance_1 = f["distance"][()]
                    angle_1 = f['DoD_phi'][()][0]


                random_BS_2 = self.BS_list[random.randint(0, len(self.BS_list)-1)]
                random_BW_2 = self.BW_list[random.randint(0, len(self.BW_list)-1)]
                if (random_BS_1==random_BS_2) & (random_BW_1==random_BW_2):
                    random_BS_2 = self.BS_list[random.randint(0, len(self.BS_list)-1)]
                    random_BW_2 = self.BW_list[random.randint(0, len(self.BW_list)-1)]
                if (random_BS_1==random_BS_2) & (random_BW_1==random_BW_2):
                    continue
                data_path = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/DeepMIMO/O1_3p5/"
                with h5py.File(data_path+f"BS{random_BS_2}_128sc_32at_{random_BW_2}M"+f"/{random_BS_2}_{random_number}.h5py", 'r', swmr=True) as f:
                    channel_real = f["channel_real"][:]
                    channel_imag = f["channel_imag"][:]
                    channel_2 = channel_real+ 1j* channel_imag
                    UElocation_2 = f["UElocation"][:][:2]
                    BSlocation_2 = f["BSlocation"][:][:2]
                    distance_2 = f["distance"][()]
                    angle_2 = f['DoD_phi'][()][0]
                if (np.sum(np.abs(channel_1) ** 2) >0) & (np.sum(np.abs(channel_2) ** 2) >0):
                    break

            # channel normalization
            channel_1 = channel_normalization(channel_1)
            # snr_db = random.randint(0, 40)
            # channel_1 = channel_normalization_with_noise(channel_1, snr_db)

            # CFR with amplitude and phase 
            channel_aa_1 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
            channel_aa_1[0,:,:] = np.abs(channel_1).astype(np.float32)
            channel_aa_1[1,:,:] = np.angle(channel_1).astype(np.float32)
            # CFR with real and imaginary
            channel_ri_1 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
            channel_ri_1[0,:,:] = np.real(channel_1).astype(np.float32)
            channel_ri_1[1,:,:] = np.imag(channel_1).astype(np.float32)
            channel_delay_angle_1 = freq_to_delay_angle_domain(channel_ri_1)
            # delay angle domain data with real and imaginary
            channel_delay_angle_ri_1 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
            channel_delay_angle_ri_1[0,:,:] = np.real(channel_delay_angle_1).astype(np.float32)
            channel_delay_angle_ri_1[1,:,:] = np.imag(channel_delay_angle_1).astype(np.float32)


            ########## position channel ##########
            # channel normalization  
            channel_2 = channel_normalization(channel_2)
            # snr_db = random.randint(0, 40)
            # channel_1 = channel_normalization_with_noise(channel_1, snr_db)

            # CFR with amplitude and phase 
            channel_aa_2 = np.zeros([self.input_fmap, channel_2.shape[1], channel_2.shape[2]])
            channel_aa_2[0,:,:] = np.abs(channel_2).astype(np.float32)
            channel_aa_2[1,:,:] = np.angle(channel_2).astype(np.float32)
            # CFR with real and imaginary
            channel_ri_2 = np.zeros([self.input_fmap, channel_2.shape[1], channel_2.shape[2]])
            channel_ri_2[0,:,:] = np.real(channel_2).astype(np.float32)
            channel_ri_2[1,:,:] = np.imag(channel_2).astype(np.float32)

            # delay angle domain data with real and imaginary
            channel_delay_angle_2 = freq_to_delay_angle_domain(channel_ri_2)
            channel_delay_angle_ri_2 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
            channel_delay_angle_ri_2[0,:,:] = np.real(channel_delay_angle_2).astype(np.float32)
            channel_delay_angle_ri_2[1,:,:] = np.imag(channel_delay_angle_2).astype(np.float32)
          

            # configurations 0,1:BSlocation, 2:Bandwidth
            BSconf_1 = np.zeros((1,3))[0,:]
            BSconf_1[:2] = BSlocation_1 / 100
            BSconf_1[2] = random_BW_1 / 10
            BSconf_2 = np.zeros((1,3))[0,:]
            BSconf_2[:2] = BSlocation_2 / 100
            BSconf_2[2] = random_BW_2 / 10    
            UElocation_all = np.zeros([2,4])
            BSconf_all = np.zeros([2,3])        

            # spatial-frequency domain data with amplitude and phase
            channel_Antenna_subcarrier_aa[0,:] = channel_aa_1
            channel_Antenna_subcarrier_aa[1,:] = channel_aa_2
            # spatial-frequency domain data with real and imaginary
            channel_Antenna_subcarrier_ri[0,:] = channel_ri_1
            channel_Antenna_subcarrier_ri[1,:] = channel_ri_2
            # delay angle domain data with real and imaginary
            channel_delay_angle_ri[0,:] = channel_delay_angle_ri_1
            channel_delay_angle_ri[1,:] = channel_delay_angle_ri_2

            # labels: 0,1:location, 2:distance(toa), 3:angle (aoa)
            UElocation_all[0,:2] = UElocation_1 /10
            UElocation_all[1,:2] = UElocation_2 /10
            UElocation_all[0, 2] = distance_1 /100   
            UElocation_all[1, 2] = distance_2 /100
            UElocation_all[0, 3] = angle_1 /100   
            UElocation_all[1, 3] = angle_2 /100

            BSconf_all[0,:] = BSconf_1
            BSconf_all[1,:] = BSconf_2
                
        return  channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all #, mask[0,:].float()
    





class generate_Dataset_test(Dataset):
    def __init__(self, EnvPara):
        
        self.input_fmap = EnvPara["input_fmap"]
        self.input_tdim = EnvPara["input_tdim"]
        self.input_fdim = EnvPara["input_fdim"]
        self.task = EnvPara["task"]
        self.EnvPara = EnvPara
        if self.task == "inference_SingleBSLoc":
            self.data_list = np.load("../pre_data/test_10000.npy")
        elif self.task == "inference_multiBSLoc":
            self.data_list = np.load("../pre_data/test_10000.npy")
        elif self.task == "inference_cnn":
            self.data_list = np.load("../pre_data/test_10000.npy")   
        elif self.task == "inference_aoa":
            self.data_list = np.load("../pre_data/test_10000.npy")  
        elif self.task == "inference_toa":
            self.data_list = np.load("../pre_data/test_10000.npy")                         
        else:
            self.data_list = np.load("../pre_data/pretrain_data.npy")

        self.len = len(self.data_list)
        print("data_list",len(self.data_list))
        self.BS_list = [3, 4, 9 ,10]
        self.BW_list = [10, 20, 50]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        file_number = self.data_list[idx]


        random_BW_1 = self.EnvPara["BW"] 
        data_path = "/mimer/NOBACKUP/groups/e2e_comms/guangjin/DeepMIMO/O1_3p5/"
        channel_data_aa = np.zeros((self.EnvPara["BS_Num"],self.input_fmap, self.input_tdim, self.input_fdim))
        channel_Antenna_subcarrier_aa = np.zeros((self.EnvPara["BS_Num"], self.input_fmap, self.input_tdim, self.input_fdim))
        channel_Antenna_subcarrier_ri = np.zeros((self.EnvPara["BS_Num"], self.input_fmap, self.input_tdim, self.input_fdim))
        channel_delay_angle_ri = np.zeros((self.EnvPara["BS_Num"], self.input_fmap, self.input_tdim, self.input_fdim))        
        UElocation_all = np.zeros([self.EnvPara["BS_Num"],4])
        BSconf_all = np.zeros([self.EnvPara["BS_Num"],3])
        flag = 0
        # for BS_i in range(len(self.BS_list)):
        for BS_i in range(self.EnvPara["BS_Num"]):
            random_BS_1 = self.BS_list[BS_i]
            with h5py.File(data_path+f"BS{random_BS_1}_128sc_32at_{random_BW_1}M"+f"/{random_BS_1}_{file_number}.h5py", 'r') as f:
                channel_real = f["channel_real"][:]
                channel_imag = f["channel_imag"][:]
                channel_1 = channel_real+ 1j* channel_imag
                UElocation_1 = f["UElocation"][:][:2]
                BSlocation_1 = f["BSlocation"][:][:2]
                distance_1 = f["distance"][()]
                angle_1 = f['DoD_phi'][()][0]
            if np.sum(np.abs(channel_1) ** 2) == 0:
                UElocation_1[:] = 0
                flag = 1

            channel_1 = channel_normalization(channel_1)
            # snr_db = 10 # random.randint(0, 40)
            # channel_1 = channel_normalization_with_noise(channel_1, snr_db)
            channel_aa_1 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
            channel_aa_1[0,:,:] = np.abs(channel_1).astype(np.float32)
            channel_aa_1[1,:,:] = np.angle(channel_1).astype(np.float32)
            channel_ri_1 = np.zeros([self.input_fmap, channel_1.shape[1], channel_1.shape[2]])
            channel_ri_1[0,:,:] = np.real(channel_1).astype(np.float32)
            channel_ri_1[1,:,:] = np.imag(channel_1).astype(np.float32)   
            channel_delay_angle_1 = freq_to_delay_angle_domain(channel_ri_1)
            channel_delay_angle_ri_1 = np.zeros([self.input_fmap, channel_ri_1.shape[1], channel_ri_1.shape[2]])
            channel_delay_angle_ri_1[0,:,:] = np.real(channel_delay_angle_1).astype(np.float32)
            channel_delay_angle_ri_1[1,:,:] = np.imag(channel_delay_angle_1).astype(np.float32)

            channel_Antenna_subcarrier_aa[BS_i,:] = channel_aa_1.copy()
            channel_Antenna_subcarrier_ri[BS_i,:] = channel_ri_1.copy()
            channel_delay_angle_ri[BS_i,:] = channel_delay_angle_ri_1.copy()
            UElocation_all[BS_i,:2] = UElocation_1.copy() / 10 
            UElocation_all[BS_i, 2] = distance_1/100 
            UElocation_all[BS_i, 3] = angle_1/100 
            if flag ==1:
                UElocation_all[BS_i,:2] = 0
            BSconf_all[BS_i,:2] = BSlocation_1.copy() / 100  
            BSconf_all[BS_i, 2] = random_BW_1 / 10


        return  channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all 

