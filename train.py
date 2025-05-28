    
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
import argparse


def set_seed(seed=42):
    random.seed(seed)  # Python 
    np.random.seed(seed)  # NumPy 
    torch.manual_seed(seed)  # PyTorch CPU 
    torch.cuda.manual_seed(seed)  # PyTorch GPU 
    torch.cuda.manual_seed_all(seed) 


parser = argparse.ArgumentParser(description="Environment Parameters")

# Training Parameters
parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--input_tdim", type=int, default=32)    # attenna number
parser.add_argument("--input_fdim", type=int, default=128)   # CSI feature map
parser.add_argument("--input_fmap", type=int, default=2)     # subcarrier number
parser.add_argument("--fshape", type=int, default=4)         # patch parametrs
parser.add_argument("--tshape", type=int, default=4)         # patch parametrs
parser.add_argument("--fstride", type=int, default=4)        # patch parametrs
parser.add_argument("--tstride", type=int, default=4)        # patch parametrs
parser.add_argument("--task", type=str, default='SingleBSLoc')
parser.add_argument("--model_path", type=str, default='./model/SingleBSLoc/')
parser.add_argument("--load_pretrained_mdl_path", type=str, default='./pretrained_model/pretrain_mix/testepoch=200.ckpt')
parser.add_argument("--pretrain_stage", type=bool, default=True)
parser.add_argument("--embed_dim", type=int, default=256)
parser.add_argument("--DTI_embed_dim", type=int, default=96)
parser.add_argument("--AFMCM_embed_dim", type=int, default=96)
parser.add_argument("--PICL_embed_dim", type=int, default=64)

parser.add_argument("--depth", type=int, default=4)            # transformer parametrs
parser.add_argument("--latent_dim", type=int, default=128)     # transformer parametrs 
parser.add_argument("--num_heads", type=int, default=4)        # transformer parametrs 
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--mask_antenna_number", type=int, default=16)
parser.add_argument("--mask_subcarrier_number", type=int, default=64)
parser.add_argument("--is_frozen", type=int, default=1)
parser.add_argument("--BW", type=int, default=10)
parser.add_argument("--DTI_flag", type=int, default=1)
parser.add_argument("--AFMCM_flag", type=int, default=1)
parser.add_argument("--PICL_flag", type=int, default=1)
parser.add_argument("--FT_dataset", type=int, default=10000)
parser.add_argument("--pilot_subcarrier_interval", type=int, default=1) # pilot interval: 1 means all channel are used as pilots
parser.add_argument("--pilot_antenna_interval", type=int, default=1)    # pilot interval: 1 means all channel are used as pilots
parser.add_argument("--BS_Num", type=int, default=4)
parser.add_argument("--is_load", type=int, default=0)
args = parser.parse_args()


print(args)
EnvPara = {
    "epochs": args.epochs,
    "input_tdim": args.input_tdim,
    "input_fdim": args.input_fdim,
    "input_fmap": args.input_fmap, 
    "fshape": args.fshape,
    "tshape": args.tshape,
    "fstride": args.fstride,
    "tstride": args.tstride,
    "task": args.task,
    "model_path": args.model_path,
    "load_pretrained_mdl_path": args.load_pretrained_mdl_path,
    "pretrain_stage": args.pretrain_stage,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "embed_dim": args.embed_dim,
    "depth": args.depth,
    "latent_dim": args.latent_dim,
    "num_heads": args.num_heads,
    "lr": args.lr,
    "mask_antenna_number": args.mask_antenna_number,
    "mask_subcarrier_number": args.mask_subcarrier_number,
    "is_frozen": args.is_frozen,
    "BW": args.BW,
    "DTI_embed_dim": args.DTI_embed_dim,
    "AFMCM_embed_dim": args.AFMCM_embed_dim,
    "PICL_embed_dim": args.PICL_embed_dim,
    "DTI_flag": args.DTI_flag,
    "AFMCM_flag": args.AFMCM_flag,
    "PICL_flag": args.PICL_flag,
    "FT_dataset": args.FT_dataset,
    "pilot_subcarrier_interval": args.pilot_subcarrier_interval,
    "pilot_antenna_interval": args.pilot_antenna_interval,
    "BS_Num": args.BS_Num,
    "is_load": args.is_load
}


EnvPara["input_feature_dim"] = 2 *  EnvPara["BS_Num"]
print("EnvPara[input_feature_dim]",EnvPara["input_feature_dim"])

# Set environment variable
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == '__main__':   
    print(EnvPara)

    FMmodel = Wrapper(EnvPara=EnvPara)

    if (EnvPara["task"] == "SingleBSLoc") | (EnvPara["task"] == "MultiBSLoc") | (EnvPara["task"] == "aoa") | (EnvPara["task"] == "toa") | (EnvPara["is_load"] == 1):
        FMmodel = Wrapper.load_from_checkpoint(EnvPara["load_pretrained_mdl_path"], EnvPara=EnvPara, strict=False)
        print("load:", EnvPara["load_pretrained_mdl_path"])

    FMmodel.to(EnvPara["device"])

    train_dataset = generate_Dataset(EnvPara)
    train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True, drop_last=False, pin_memory=True)
    val_dataset = generate_Dataset(EnvPara, is_val=1)
    val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8, shuffle=True, drop_last=False, pin_memory=True)

    model_path = "test"
    model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename=model_path + '{epoch:02d}',
        save_top_k=1,  # Only save the best model
        monitor="val/ave_loss",  # Monitor validation loss to determine the best model
        mode="min",  # 'min' means the lower the better; use 'max' for metrics like accuracy
        save_weights_only=True,
    )

    latest_model_checkpoint = ModelCheckpoint(
        dirpath=EnvPara["model_path"],
        filename=model_path + '_latest',
        save_top_k=1,  # Only keep the latest model
        save_last=True,  # Always save the most recent model
        every_n_epochs=5,  # Save every n epochs
        save_weights_only=True,
    )

    logger = pl.loggers.TensorBoardLogger('./logs', name=EnvPara["task"])
    trainer = Trainer(
        log_every_n_steps=0,  # Do not log at each step
        enable_progress_bar=False,
        # precision=16,
        devices=1,
        accelerator='gpu',
        logger=logger,
        max_epochs=EnvPara["epochs"],
        callbacks=[model_checkpoint, latest_model_checkpoint],
    )
    trainer.fit(FMmodel, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
