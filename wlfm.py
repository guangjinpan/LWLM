import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from random import randrange
from torchvision.models import resnet34

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)    


def cosine_similarity_loss(H1, H2, eps = 1e-1 ):


    H1r = H1[:,0,:,:] 
    H1i = H1[:,1,:,:]  
    H2r = H2[:,0,:,:]  
    H2i = H2[:,1,:,:] 
    B, M, N = H1r.shape
    # 1) reshape to [B, M*N]
    xi = H1r.reshape(B,-1)  # x real
    xj = H1i.reshape(B,-1)  # x imag
    yi = H2r.reshape(B,-1)  # y real
    yj = H2i.reshape(B,-1)  # y imag

    # 2) calculate dot product
    #    xy_dot_r = Σ(xr * yr + xi * yi)
    #    xy_dot_i = Σ(xr * yi - xi * yr)
    xy_dot_r = torch.sum(xi * yi + xj * yj, dim=1)
    xy_dot_i = torch.sum(xi * yj - xj * yi, dim=1)

    # 3) calculate ||x|| and ||y||
    norm_x = torch.sqrt(torch.sum(xi**2, dim=1) + torch.sum(xj**2, dim=1))
    norm_y = torch.sqrt(torch.sum(yi**2, dim=1) + torch.sum(yj**2, dim=1))

    # 5) calculate sqrt( real^2 + imag^2 ) / (||x|| * ||y||)
    dot_mod = torch.sqrt(xy_dot_r**2 + xy_dot_i**2)
    cos_loss = 1- dot_mod / (norm_x * norm_y + eps)

    return torch.mean(cos_loss)



class CNNnet(nn.Module):
    """
    input  : [B, M, 128, 32] 
    output  : [B, 2]
    """
    def __init__(self, input_feature_dim):
        super().__init__()

        self.net  = resnet34()    # torchvision>=0.13

        self.net.conv1 = nn.Conv2d(input_feature_dim, 64,
                                   kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.net.conv1.weight, mode="fan_out", nonlinearity="relu")


        # MLP layer
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, x):
        return self.net(x)





class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: [B, N, D]
        attn_scores = self.attn(x).squeeze(-1)  # [B, N]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N]
        fused = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, D]
        return fused, attn_weights

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
         (B, in_chans, H, W) -> (B, num_patches, embed_dim)
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, in_chans, H, W)
        output: (B, num_patches, embed_dim)
        """
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoidal Position Encoding: (1, n_position, d_hid)"""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


# ====================== Decoder for pretrain ====================== #
class SimpleDecoder(nn.Module):

    def __init__(self, latent_dim=512, embed_dim=512,
                 depth=2, num_heads=8, dim_feedforward=2048, dropout=0.1,
                 final_out_dim=256  # 
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # latent_dim -> embed_dim
        self.latent_to_embed = nn.Linear(latent_dim, embed_dim, bias=True)

        #  TransformerDecoder 
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.proj_out = nn.Linear(embed_dim, final_out_dim)


        self.mask_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.xavier_normal_(self.mask_token)




    def forward(self, memory, tgt=None, mask_idx=None):
        B, N, _ = memory.shape
        # latent->embed 
        memory_embed = self.latent_to_embed(memory)  # (B, N, embed_dim)

        if tgt is None:
            tgt_embed = memory_embed.clone()
        else:
            tgt_embed = tgt  # 


        if mask_idx is not None:
            mask_token_embed = self.latent_to_embed(self.mask_token)  # (1,1,embed_dim)
            for i in range(B):
                idx_i = mask_idx[i]
                tgt_embed[i, idx_i, :] = mask_token_embed[0,0,:]

        # TransformerDecoder
        out = self.transformer_decoder(tgt_embed, memory_embed)  # (B, N, embed_dim)
        out = self.norm(out)
        # Projection to reconstruction dimension
        out = self.proj_out(out)  # (B, N, final_out_dim)
        return out


class wireless_loc_fm(nn.Module):

    def __init__(
        self,
        label_dim=567,
        # patch/
        fshape=4, tshape=4, fstride=4, tstride=4,
        input_fdim=64, input_tdim=32, input_fmap=2,

        # Transformer(Encoder)
        embed_dim=512,
        depth=6,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        contrast_out_dim = 32,
        BSconf_dim = 3,

        device="cpu",

        EnvPara = None,
    ):
        super().__init__()
        self.device = device
        self.EnvPara = EnvPara

        # ----------------------------------------------------
        # 1) Patch Embedding (Encoder 输入)
        # ----------------------------------------------------
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.input_fmap = input_fmap

        patch_size = (fshape, tshape)
        in_chans = input_fmap

        # patch_embed
        # ----------------------------------------------------
        # Encoder(1)：CNN layer for patch embedding
        # ----------------------------------------------------
        self.patch_embed = PatchEmbed(
            img_size=(input_fdim, input_tdim),
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # cls token (LST)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_num = 1

        # position embedding
        pos_embed = get_sinusoid_encoding(self.num_patches + self.cls_token_num, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)

        # ----------------------------------------------------
        # Encoder(2)：TransformerEncoder + LayerNorm
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        print(f"Encoder parameters: {(count_parameters(self.patch_embed)+count_parameters(self.transformer_encoder)+count_parameters(self.norm)) / 1e6:.2f}M")
        print(self.EnvPara["task"])

        if (self.EnvPara["task"] == "SingleBSLoc") | (self.EnvPara["task"] == "inference_SingleBSLoc"):
            # ----------------------------------------------------
            # Decoder(0)：Single-BS-Loc Decoder
            # ----------------------------------------------------
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.SB_positioninglayer = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2),
            )
            print(f"Single-Loc parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.SB_positioninglayer)) / 1e6:.2f}M")
        elif (self.EnvPara["task"] == "aoa") | (self.EnvPara["task"] == "inference_aoa"):
            # ----------------------------------------------------
            # Decoder：AOA Decoder
            # ----------------------------------------------------
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.aoa_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )
            print(f"aoa parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.aoa_layer)) / 1e6:.2f}M")

        elif (self.EnvPara["task"] == "toa")| (self.EnvPara["task"] == "inference_toa"):
            # ----------------------------------------------------
            # Decoder：TOA Decoder
            # ----------------------------------------------------
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.toa_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            )
            print(f"toa parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.toa_layer)) / 1e6:.2f}M")

        elif (self.EnvPara["task"] == "MultiBSLoc") | (self.EnvPara["task"] == "inference_multiBSLoc"):
            # ----------------------------------------------------
            # Decoder: Multi-BS-Loc Decoder
            # ----------------------------------------------------
            print()
            self.proj_BSconfig2 = nn.Sequential(
                nn.Linear(BSconf_dim, embed_dim),
                nn.ReLU(),
            )
            self.MB_positioninglayer0 = nn.ModuleList()
            for i in range(4):
                SB_layer = nn.Sequential(
                    nn.Linear(512, 128),
                    nn.ReLU(),
                )
                self.MB_positioninglayer0.append(SB_layer)
            self.MB_positioninglayer1 = nn.ModuleList()
            for i in range(4):
                SB_layer = nn.Sequential(
                    # nn.ReLU(),
                    nn.Linear(128, 2),
                )
                self.MB_positioninglayer1.append(SB_layer)     
            self.attn_pool2 = AttentionPooling(input_dim=128, hidden_dim=32)
            print(f"Multi-Loc parameters: {(count_parameters(self.proj_BSconfig2)+count_parameters(self.MB_positioninglayer0)) / 1e6:.2f}M")        




        self.proj_BSconfig = nn.Linear(BSconf_dim, embed_dim)
        # ----------------------------------------------------
        # Decoder：PICL Decoder
        # ----------------------------------------------------       
        self.proj_contrastive = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, contrast_out_dim)
        )
        self.proj_PICL_to_embed = nn.Linear(self.EnvPara["PICL_embed_dim"], embed_dim)

        # ----------------------------------------------------
        # Decoder：SFMCM Decoder
        # ----------------------------------------------------
        self.decoder_AFMCM_mix= SimpleDecoder(
            latent_dim=EnvPara["AFMCM_embed_dim"],
            embed_dim=embed_dim,
            depth=2,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            final_out_dim=fshape * tshape * input_fmap
        ) 
            # mask matrix
        self.mask_antenna = nn.Parameter(torch.zeros([1, input_fmap, input_fdim, input_tdim]))
        nn.init.xavier_normal_(self.mask_antenna)
        self.mask_subcarrier = nn.Parameter(torch.zeros([1, input_fmap, input_fdim, input_tdim]))
        nn.init.xavier_normal_(self.mask_subcarrier)

        self.unfold = nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))
        
        # ----------------------------------------------------
        # Decoder：DTI Decoder
        # ----------------------------------------------------
        # (B, patch_dim, N+1) => (B, in_chans, input_fdim, input_tdim)
        self.decoder_DTI_mix= SimpleDecoder(
            latent_dim=EnvPara["DTI_embed_dim"],
            embed_dim=embed_dim,
            depth=2,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            final_out_dim=fshape * tshape * input_fmap
        )           
        self.fold = nn.Fold(
            output_size=(self.input_fdim, self.input_tdim),
            kernel_size=(self.fshape, self.tshape),
            stride=(self.fstride, self.tstride)
        )


        self.L1loss = nn.L1Loss()
        if ((self.EnvPara["task"] == "CNN")):
            self.cnnnet1=CNNnet(EnvPara["input_feature_dim"])
            print(f"CNN parameters: {(count_parameters(self.cnnnet1)) / 1e6:.2f}M")


    # ================== mask ================== #
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []
        cur_clus = randrange(cluster) + 3
        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)
            cur_mask = []
            for i in range(cur_clus):
                for j in range(cur_clus):
                    mask_cand = start_id + int(np.sqrt(sequence_len)) * i + j
                    if 0 <= mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        return torch.tensor(random.sample(range(0, sequence_len), mask_size))

    def fm_encoder(self, input):
        
        B = input.shape[0]

        x = self.patch_embed(input)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.transformer_encoder(x)
        x = self.norm(x)
        return x
    
    
    # -------------- (NT-Xent Loss) --------------
    def NTXentLoss(self, z1, z2, temperature=0.1):
        """
        z1, z2: shape (B, D), 分别是同一批数据的两种增强
        return: 标量 loss
        """
        B, D = z1.shape
        z = torch.cat([z1, z2], dim=0)
        # L2 normalize => (2B, D)
        z_norm = F.normalize(z, dim=1)


        sim_matrix = torch.matmul(z_norm, z_norm.t())  # 
        
        labels = torch.arange(B, dtype=torch.long, device=z.device)
        
        mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix[~mask].view(2*B, 2*B-1)  
        sim_12 = sim_matrix[:B]  # shape (B, 2B-1)
        sim_21 = sim_matrix[B:]  # shape (B, 2B-1)
        
        sim_12 = sim_12 / temperature
        sim_21 = sim_21 / temperature
        
        loss_12 = F.cross_entropy(sim_12, labels)
        loss_21 = F.cross_entropy(sim_21, labels)
        return (loss_12 + loss_21) * 0.5
    
    
    
    def mix_pretraining(self, channel_Antenna_subcarrier_aa , channel_delay_angle_ri, BSconf_all, mask_antenna_number=8, mask_subcarrier_number = 32):

        B = channel_Antenna_subcarrier_aa.shape[0]

        x_input = channel_Antenna_subcarrier_aa[:,0,:]
        x_input_aug = channel_Antenna_subcarrier_aa[:,1,:]
        x_input_ri = channel_delay_angle_ri[:,0,:]
        x_input_ri_aug = channel_delay_angle_ri[:,1,:]
        BSconf = BSconf_all[:,0,:]
        BSconf_aug = BSconf_all[:,1,:]        


        if self.EnvPara["DTI_flag"] ==1:
            # (A) Encoder
            enc_out = self.fm_encoder(x_input)  # => (B, N+1, embed_dim)
            # Transformer-based decode
            rec = self.decoder_DTI_mix(enc_out[:,:,:self.EnvPara["DTI_embed_dim"]])  # => (B, N+1, final_out_dim)
            # Remove the cls token part => (B, num_patches, patch_dim) 
            rec_wo_cls = rec[:, self.cls_token_num:, :]   # (B, num_patches, fshape*tshape*in_chans)
            # fold => (B, in_chans, H, W)
            # transpose => (B, patch_dim, num_patches)
            rec_wo_cls = rec_wo_cls.transpose(1, 2)
            x_recon_spatial = self.fold(rec_wo_cls)
            # (H) MSE
            # loss_mse = torch.mean((x_recon_spatial - x_input_ri)**2)
            # (H) Cosine similarity
            DTI_loss_cos = cosine_similarity_loss(x_input_ri, x_recon_spatial)


        if self.EnvPara["AFMCM_flag"] ==1:
            # mask ant
            mask_index_ant = torch.empty((B, mask_antenna_number), device=self.device, dtype=torch.long)
            mask_dense_ant = torch.ones_like(x_input)
            for i in range(B):
                mask_index_ant[i] = self.gen_maskid_frame(self.input_tdim, mask_antenna_number)
                mask_dense_ant[i, :, :, mask_index_ant[i]] = 0
            mask_tokens = self.mask_antenna.expand(B, x_input.shape[1], x_input.shape[2], -1)
            x_masked = x_input * mask_dense_ant + (1 - mask_dense_ant) * mask_tokens
            # mask subcarrier
            mask_index_subc = torch.empty((B, mask_subcarrier_number), device=self.device, dtype=torch.long)
            mask_dense_subc = torch.ones_like(x_input)
            for i in range(B):
                mask_index_subc[i] = self.gen_maskid_frame(self.input_fdim, mask_subcarrier_number)
                mask_dense_subc[i, :, mask_index_subc[i], :] = 0
            mask_tokens = self.mask_antenna.expand(B, x_input.shape[1], x_input.shape[2], -1)
            x_masked = x_masked * mask_dense_subc + (1 - mask_dense_subc) * mask_tokens

            # ---- (B) Encoder ----
            enc_out = self.fm_encoder(x_masked)
            # configration embedding
            BSconf_emb = self.proj_BSconfig(BSconf)
            BSconf_emb = BSconf_emb.unsqueeze(1).expand(-1, enc_out.shape[1], -1)
            enc_out = enc_out + BSconf_emb

            # ---- (C) Decoder ----
            input_unfold = self.unfold(x_input).transpose(1, 2)
            rec = self.decoder_AFMCM_mix(enc_out[:,:,self.EnvPara["DTI_embed_dim"]:self.EnvPara["DTI_embed_dim"]+self.EnvPara["AFMCM_embed_dim"]])  # (B, N, patch_dim)
            rec_wo_cls = rec[:, self.cls_token_num:, :]  # (B, num_patches, patch_dim)
            # target
            target_wo_cls = input_unfold  # (B, num_patches, patch_dim)
            AFMCM_loss_mse = torch.mean((rec_wo_cls - target_wo_cls)**2)    

        if self.EnvPara["PICL_flag"] ==1:
            # mask mask_antenna  
            # --- (A).Encoder for x ---
            x_e = self.fm_encoder(x_input)
            # --- (B). Encoder for x_aug ---
            x_aug_e = self.fm_encoder(x_input_aug)
            # --- (C). Pooling ---
            x_e = self.proj_PICL_to_embed(x_e[:,:,self.EnvPara["DTI_embed_dim"]+self.EnvPara["AFMCM_embed_dim"]:])
            x_aug_e = self.proj_PICL_to_embed(x_aug_e[:,:,self.EnvPara["DTI_embed_dim"]+self.EnvPara["AFMCM_embed_dim"]:])

            z_x = torch.mean(x_e, dim=1)      # (B, latent_dim)
            z_x_aug = torch.mean(x_aug_e, dim=1)# (B, latent_dim)
            z_x_c = self.proj_BSconfig(BSconf)
            z_x_c_aug  = self.proj_BSconfig(BSconf_aug)
            z_x = z_x_c + z_x
            z_x_aug = z_x_c_aug + z_x_aug
            # --- (D). projector ---
            z1 = self.proj_contrastive(z_x)       # (B, contrast_out_dim)
            z2 = self.proj_contrastive(z_x_aug)   # (B, contrast_out_dim)
            # --- (E). compute NT-Xent loss ---
            PICL_loss_mse = self.NTXentLoss(z1, z2, temperature=0.2)                        

        return 20*DTI_loss_cos +  10*AFMCM_loss_mse + PICL_loss_mse


    def toa_est(self, channel_data, distance, BSconf_all): 
        input = channel_data[:,0,:]
        BSconf = BSconf_all[:,0,:]
        B = input.shape[0]
        distance = distance[:,0].reshape((B,1))
        N_a = input.shape[3]
        N_s = input.shape[2] 

        mask_dense_ant_subc = 1-torch.ones_like(input)
        for i in range(0,N_a,self.EnvPara["pilot_antenna_interval"]):
            for j in range(0,N_s,self.EnvPara["pilot_subcarrier_interval"]):
                mask_dense_ant_subc[:, :, j, i] = 1
        input = input * mask_dense_ant_subc  

        if self.EnvPara["is_frozen"]==1:
            with torch.no_grad():
                x = self.fm_encoder(input)
        else:
            x = self.fm_encoder(input)  

        x_mean = x[:,0,:].view((B,-1))
        z_x_c = self.proj_BSconfig2(BSconf)
        z_x_list=[]
        z_x_list.append(x_mean)
        z_x_list.append(z_x_c)
        z_x_concat = torch.cat(z_x_list, dim=1)
        pred = self.toa_layer(z_x_concat)
        loss = self.L1loss(pred, distance)
        return pred, loss
    
    def aoa_est(self, channel_data, angle, BSconf_all): 
        input = channel_data[:,0,:]
        
        BSconf = BSconf_all[:,0,:]
        B = input.shape[0]
        angle = angle[:,0].reshape((B,1))
        N_a = input.shape[3]
        N_s = input.shape[2] 

        mask_dense_ant_subc = 1-torch.ones_like(input)
        for i in range(0,N_a,self.EnvPara["pilot_antenna_interval"]):
            for j in range(0,N_s,self.EnvPara["pilot_subcarrier_interval"]):
                mask_dense_ant_subc[:, :, j, i] = 1
        input = input * mask_dense_ant_subc  

        # --- A. Encoder for x ---
        if self.EnvPara["is_frozen"]==1:
            with torch.no_grad():
                x = self.fm_encoder(input)
        else:
            x = self.fm_encoder(input)   
        x_mean = x[:,0,:].view((B,-1))
        z_x_c = self.proj_BSconfig2(BSconf)
        z_x_list=[]
        z_x_list.append(x_mean)
        z_x_list.append(z_x_c)
        z_x_concat = torch.cat(z_x_list, dim=1)
        pred = self.aoa_layer(z_x_concat)
        loss = self.L1loss(pred, angle)
        return pred, loss

    
    def singleBSLoc_with_pilot(self, channel_data, UElocation_all, BSconf_all): 
        input = channel_data[:,0,:]
        y_position = UElocation_all[:,0,:]
        BSconf = BSconf_all[:,0,:]
        B = input.shape[0]
        N_a = input.shape[3]
        N_s = input.shape[2] 

        mask_dense_ant_subc = 1-torch.ones_like(input)
        for i in range(0,N_a,self.EnvPara["pilot_antenna_interval"]):
            for j in range(0,N_s,self.EnvPara["pilot_subcarrier_interval"]):
                mask_dense_ant_subc[:, :, j, i] = 1
        input = input * mask_dense_ant_subc  

        # --- A. Encoder for x ---
        if self.EnvPara["is_frozen"] == 1:
            modules_to_freeze = [
                self.patch_embed,
                self.transformer_encoder,
                self.norm
            ]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
            self.pos_embed.requires_grad = False
            self.cls_token.requires_grad = False


        x = self.fm_encoder(input)   
        x_mean = x[:,0,:].view((B,-1))
        z_x_c = self.proj_BSconfig2(BSconf)
        z_x_list=[]
        z_x_list.append(x_mean)
        z_x_list.append(z_x_c)
        z_x_concat = torch.cat(z_x_list, dim=1)
        pred = self.SB_positioninglayer(z_x_concat)
        dis = torch.sum((pred - y_position) ** 2, 1)
        mse = torch.mean(torch.sqrt(dis))
        # mse = torch.sqrt(torch.mean((pred - y_position) ** 2))
        
        return pred, mse 
        
    

    

    def multiBSLoc_att(self, channel_data, UElocation_all, BSconf_all, mode = "train"):
        y_position = UElocation_all[:, 0, :]
        B = channel_data.shape[0]
        if mode == "train":
            T = random.randint(2, 4) 
        else:
            T = self.EnvPara["BS_Num"]

        emb_list1 = []

        for i in range(T):
            input = channel_data[:, i, :]
            BSconf = BSconf_all[:, i, :] 

            if self.EnvPara["is_frozen"]==1:
                with torch.no_grad():
                    x = self.fm_encoder(input)
            else:
                x = self.fm_encoder(input)
            x_mean = x[:,0,:].view((B,-1))
            z_x_c = self.proj_BSconfig2(BSconf)
            z_x_list = []
            z_x_list.append(x_mean)
            z_x_list.append(z_x_c)
            z_x_concat = torch.cat(z_x_list, dim=1)
            pred = self.MB_positioninglayer0[i](z_x_concat)
            emb_list1.append(pred)

        bs_embeddings0 = torch.stack(emb_list1, dim=1)
        att_embed, attn_weights = self.attn_pool2(bs_embeddings0)
        emb_list2  = []
        for i in range(T):
            bs_embeddings_i = bs_embeddings0[:, i, :]
            pred = self.MB_positioninglayer1[i](bs_embeddings_i)
            emb_list2.append(pred)
        
        bs_embeddings1 = torch.stack(emb_list2, dim=1)
        pred_res = torch.bmm(attn_weights.unsqueeze(1), bs_embeddings1).squeeze(1)  # [B, D]
        dis = torch.sum((pred_res - y_position) ** 2, 1)
        mse = torch.mean(torch.sqrt(dis))*1

        if mode == "train":
            for i in range(T):
                dis1 = torch.sum((bs_embeddings1[:,i,:] - y_position) ** 2, 1)
                mse = mse + torch.mean(torch.sqrt(dis1))*0.1
        # mse = torch.sqrt(torch.mean((pred - y_position) ** 2))
        return pred_res, mse 
        




    def forward(self, channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, channel_delay_angle_ri, UElocation_all, BSconf_all, task = "pretrain_mix",
                cluster=True, mask_antenna_number=8, mask_subcarrier_number = 32):
        """
        x: (B, input_fmap, input_fdim, input_tdim)
        task: 
            - "SingleBSLoc", "inference_SingleBSLoc"
            - "MultiBSLoc", "inference_MultiBSLoc"
            - "SingleBSLoc", "inference_SingleBSLoc"
            - "toa", "inference_toa"
            - "aoa", "inference_aoa"
            - "pretrain_mix" 
        """
        #x, y_position, BSconf = None, x_aug = None, y_position_aug = None, BSconf_aug = None
        # x = channel_data, UElocation, BSconf
        # x = x.transpose(2, 3)  # [B, in_chans, H, W]
        # if x_aug !=None:
        #     x_aug = x_aug.transpose(2, 3)
        # torch.autograd.set_detect_anomaly(True)

        channel_Antenna_subcarrier_aa = channel_Antenna_subcarrier_aa.transpose(3, 4)
        channel_Antenna_subcarrier_ri = channel_Antenna_subcarrier_ri.transpose(3, 4)
        channel_delay_angle_ri = channel_delay_angle_ri.transpose(3, 4)
        distance = UElocation_all[:,:,2]
        angle = UElocation_all[:,:,3]
        UElocation_all = UElocation_all[:,:,:2]


        if task == "SingleBSLoc":
            pred, mse = self.singleBSLoc_with_pilot(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all)
            return mse
        elif task == "inference_SingleBSLoc":
            return self.singleBSLoc_with_pilot(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all)
        elif task == "MultiBSLoc":
            pred, loss =  self.multiBSLoc_att(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all, mode ="train")
            return loss
        elif task == "inference_multiBSLoc":
            return self.multiBSLoc_att(channel_Antenna_subcarrier_aa, UElocation_all, BSconf_all, mode = "test")
        elif task == "toa":
            pred, loss = self.toa_est(channel_Antenna_subcarrier_aa, distance, BSconf_all)  
            return loss  
        elif task == "inference_toa":
            return self.toa_est(channel_Antenna_subcarrier_aa, distance, BSconf_all)     
        elif task == "aoa":
            pred, loss = self.aoa_est(channel_Antenna_subcarrier_aa, angle, BSconf_all)
            return loss
        elif task == "inference_aoa":
            return self.aoa_est(channel_Antenna_subcarrier_aa, angle, BSconf_all)
        
        # ============== pretrain task ==============
        elif task == "pretrain_mix": 
            return self.mix_pretraining(channel_Antenna_subcarrier_aa, channel_Antenna_subcarrier_ri, mask_antenna_number=mask_antenna_number, mask_subcarrier_number=mask_subcarrier_number, BSconf_all = BSconf_all)   
        else:
            raise ValueError(f"Unsupported task: {task}")


