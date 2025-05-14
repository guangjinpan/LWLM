# LWLM: Large Wireless Localization Model

This repository contains the implementation of the paper:  
**"Large Wireless Localization Model (LWLM): A Foundation Model for Positioning in 6G Networks."**  

---

## 🔧 How to Run

### 🏋️‍♂️ Pretraining

To pretrain the model, run:

```bash
python train.py \
  --task pretrain_mix \
  --model_path ../../pretrained_model/testmix \
  --is_frozen 0 \
  --is_load 0 \
  --depth 4 \
  --load_pretrained_mdl_path ../../pretrained_model/pretrain_mix/testepoch=139.ckpt


### 🎯 Fine-tuning

To fine-tune the pretrained model, run:

```bash
python train.py \
  --task SingleBSLoc \
  --model_path ../../pretrained_model/SingleBSLoc/mix/nofrozen/10000/test2 \
  --is_frozen 0 \
  --is_load 1 \
  --depth 4 \
  --load_pretrained_mdl_path ../../pretrained_model/pretrain_mix/testepoch=139.ckpt \
  --FT_dataset 10000 \
  --BW 10 \
  --pilot_subcarrier_interval 1 \
  --pilot_antenna_interval 1 \
  --BS_Num 4


##  If you find this work helpful, please consider citing:


