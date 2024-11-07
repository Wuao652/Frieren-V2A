import os
import sys
import random
import argparse

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join("/".join(os.getcwd().split("/")[:-1]), "Frieren"))
from cfm.util import instantiate_from_config

class CavpDataset(Dataset):
    def __init__(self, cavp_files, truncate_len=32, device=None):
        self.cavp_files = cavp_files
        self.truncate_len = truncate_len
        self.device = device

    def __getitem__(self, index):
        cavp_file = self.cavp_files[index]
        assert os.path.exists(cavp_file), f'{cavp_file} does not exist'
        try:
            video_feat = np.load(cavp_file)
        except Exception as e:
            print('Error:', e)
        if cavp_file.endswith('.npz'):
            video_feat = video_feat['feat']
        video_feat = torch.Tensor(video_feat)
        if video_feat.shape[0] >= self.truncate_len:
            video_feat = video_feat[:self.truncate_len]
        else:
            video_feat = torch.nn.functional.pad(video_feat, (0, 0, 0, self.truncate_len - video_feat.shape[0]), 'constant')
        
        return (video_feat, cavp_file)

    def __len__(self):
        return len(self.cavp_files)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def generate_mel(ldm_config_path, ldm_ckpt_path, cavp_feature_list, cavp_feature_dir, gpus, out_dir, timesteps, solver, cfg_scale):
    # Set Device:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = torch.device("cuda")
    
    seed_everything(21)

    batch_size = 1
    steps = timesteps                # Inference Steps
    sampler = solver    # DPM-Solver Sampler
    truncate_len = 32

    os.makedirs(out_dir, exist_ok=True)

    # LDM Config:
    config = OmegaConf.load(ldm_config_path)

    # Loading LDM:
    latent_diffusion_model = load_model_from_config(config, ldm_ckpt_path).to(device)

    all_npy_file = open(cavp_feature_list).readlines()
    all_npy_file = [os.path.join(cavp_feature_dir, item.strip() + '.npz') for item in all_npy_file]
    all_npy = [item.strip() for item in all_npy_file]

    check_npy = []
    for npy_file in all_npy:
        try:
            np.load(npy_file)
            check_npy.append(npy_file)
        except Exception as e:
            print('Error:', e)

    dataset = CavpDataset(check_npy, truncate_len=truncate_len, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    for batch in tqdm(loader): 
        video_feats, cavp_files = batch 
        video_feats = video_feats.to(device)
        mel_path = os.path.join(out_dir, os.path.basename(cavp_files[-1]).split('.')[0] + '_mel.npy')
        if os.path.exists(mel_path):
            continue
            
        embed_cond_feat = latent_diffusion_model.get_learned_conditioning(video_feats)     
        audio_samples, _, noise = latent_diffusion_model.sample_param_noise_cfg(embed_cond_feat, cfg_scale=cfg_scale, batch_size=batch_size, solver=sampler, timesteps=steps)
        audio_samples = latent_diffusion_model.decode_first_stage(audio_samples)

        if len(audio_samples.shape) == 4:
            audio_samples = audio_samples[:, 0, :, :].detach().cpu().numpy()                               
        else:
            audio_samples = audio_samples.detach().cpu().numpy()     
        
        noise = noise.detach().cpu().numpy()    
    
        for i, mel_spec in enumerate(audio_samples):
            mel_path = os.path.join(out_dir, os.path.basename(cavp_files[i]).split('.')[0] + '_mel.npy')
            np.save(mel_path, mel_spec)

            noise_path = os.path.join(out_dir, os.path.basename(cavp_files[i]).split('.')[0] + '_noise.npy')
            np.save(noise_path, noise[i])

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--ldm',
        type=str,
        help='ldm model checkpoints path',
    )
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='ldm model config path',
    )
    parser.add_argument(
        '-g',
        '--gpus',
        type=str,
        default="0",
        help='use gpu',
    )
    parser.add_argument(
        '-f',
        '--cavp-feature-list',
        type=str,
        help='cavp item list',
    )
    parser.add_argument(
        '-d',
        '--cavp-feature-dir',
        type=str,
        help='cavp feature dir',
    )
    parser.add_argument(
        '-o',
        '--out-dir',
        type=str,
        help='gen mel save dir',
    )
    parser.add_argument(
        '-t',
        '--timesteps',
        type=int,
        default=25,
    )
    parser.add_argument(
        '-s',
        '--solver',
        type=str,
        default='euler',
    )

    parser.add_argument(
        '--scale',
        type=float,
        default=4.5,
    )
    
    args = parser.parse_args()
    generate_mel(args.config, args.ldm, args.cavp_feature_list, args.cavp_feature_dir, args.gpus, args.out_dir, args.timesteps, args.solver, args.scale)
