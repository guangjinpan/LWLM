    
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
from dataload import *


import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint


EnvPara = {}  
EnvPara["epochs"]  = 100
EnvPara["input_tdim"]  = 32
EnvPara["input_fdim"]  = 128
EnvPara["input_fmap"]  = 2
EnvPara["fshape"]  = 4
EnvPara["tshape"]  = 4
EnvPara["fstride"]  = 4
EnvPara["tstride"]  = 4
EnvPara["task"]  = "inference_SingleBSLoc"
EnvPara["load_pretrained_mdl_path"] =  './model/SingleBSLoc/testepoch=975.ckpt' 
EnvPara["save_result_file"] = 'res_mix-nofrozen-10M'
EnvPara["pretrain_stage"]  = False
EnvPara["device"]  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EnvPara["BS_Num"] = 1
EnvPara["input_feature_dim"] = EnvPara["BS_Num"] * 2

EnvPara["embed_dim"] = 256
EnvPara["depth"] = 4
EnvPara["latent_dim"] =128
EnvPara["num_heads"] = 4
EnvPara["is_frozen"] = 0
EnvPara["BW"] = 10
EnvPara["pilot_antenna_interval"]=1
EnvPara["pilot_subcarrier_interval"]=1

EnvPara["DTI_embed_dim"] = 96
EnvPara["AFMCM_embed_dim"] = 96
EnvPara["PICL_embed_dim"] = 64

EnvPara["DTI_flag"] = 1
EnvPara["AFMCM_flag"] = 1
EnvPara["PICL_flag"] = 1


if __name__ == '__main__':   

    print(EnvPara)

    result = np.zeros((500000,2))
    label = np.zeros((500000,2))
    mse = np.zeros((500000,1))
    cnt = 0
    UElocation_est = np.zeros((1,2))

    FMmodel = Wrapper(EnvPara)
    FMmodel= Wrapper.load_from_checkpoint(EnvPara["load_pretrained_mdl_path"], EnvPara=EnvPara, strict=False)
    FMmodel.to(EnvPara["device"])
    test_dataset = generate_Dataset_test(EnvPara)
    test_dataloader = DataLoader(test_dataset, batch_size = 32, num_workers = 8, shuffle = False, drop_last = False, pin_memory=True)

    for  channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all in test_dataloader:
        FMmodel.eval()
        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.to(EnvPara["device"]).float()
        UElocation_all = UElocation_all.to(EnvPara["device"]).float()
        BSconf_all = BSconf_all.to(EnvPara["device"]).float()
        pred, loss = FMmodel.channel_fdmdl(channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri, channel_delay_angle_ri = channel_delay_angle_ri, UElocation_all = UElocation_all, BSconf_all = BSconf_all, task = EnvPara["task"])
        result[cnt : cnt+len(channel_Antenna_subcarrier_aa)] = pred.cpu().detach().numpy() * 10
        label[cnt : cnt+len(channel_Antenna_subcarrier_aa)] = UElocation_all[:,0,:2].cpu().detach().numpy() * 10
        mse[cnt : cnt+len(channel_Antenna_subcarrier_aa),0] = loss.cpu().detach().numpy()
        cnt = cnt+len(channel_Antenna_subcarrier_aa) 

    # delete the non-coverage data
    valid_indices = np.where(np.sum(np.abs(label),1) != 0)[0]
    valid_result = result[valid_indices,:]
    valid_label = label[valid_indices,:]
    valid_mes= mse[valid_indices,:]

    # save results as txt
    merged = np.hstack((valid_label, valid_result))
    np.savetxt("./result/"+EnvPara["save_result_file"]+".txt", merged, fmt="%.6f", delimiter=' ')

    #compute distances
    distances = np.linalg.norm(valid_result - valid_label, axis=1)


    # Compute mean error
    mean_error = np.mean(np.sqrt(distances ** 2))

    print(f"mean error: {mean_error}, loss:{np.mean(valid_mes)}")