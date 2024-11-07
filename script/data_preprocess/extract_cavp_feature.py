import importlib
from omegaconf import OmegaConf
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

import cv2
import torchvision.transforms as transforms
from PIL import Image

def instantiate_from_config(config,reload=False):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"],reload=reload)(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Extract_CAVP_Features(torch.nn.Module):

    def __init__(self, fps=4, batch_size=2, device=None,  video_shape=(224,224), config_path=None, ckpt_path=None):
        super(Extract_CAVP_Features, self).__init__()
        self.fps = fps
        self.batch_size = batch_size
        self.device = device

        # Initalize Stage1 CAVP model:
        print("Initalize Stage1 CAVP Model")
        config = OmegaConf.load(config_path)
        self.stage1_model = instantiate_from_config(config.model).to(device)

        # Loading Model from:
        assert ckpt_path is not None
        print("Loading Stage1 CAVP Model from: {}".format(ckpt_path))
        self.init_first_from_ckpt(ckpt_path)
        self.stage1_model.eval()
        
        # Transform:
        self.img_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor(),
        ])
    
    
    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.stage1_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    
    @torch.no_grad()
    def forward(self, video_path_low_fps, start_second=None, truncate_second=None):
        extraction_fps = self.fps

        # read the video:
        cap = cv2.VideoCapture(video_path_low_fps)

        feat_batch_list = []
        video_feats = []
        first_frame = True
        # pbar = tqdm(cap.get(7))
        i = 0
        while cap.isOpened():
            i += 1
            # pbar.set_description("Processing Frames: {} Total: {}".format(i, cap.get(7)))
            frames_exists, rgb = cap.read()
            
            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
                
                # Forward:
                if len(feat_batch_list) == self.batch_size:
                    # Stage1 Model:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                    feat_batch_list = []
            else:
                if len(feat_batch_list) != 0:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                cap.release()
                break
        
        video_contrastive_feats = np.concatenate(video_feats)
        
        return video_contrastive_feats

if __name__ == '__main__':
    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument('--item-list-path', type=str)
    parser.add_argument('--cavp-config-path', type=str)
    parser.add_argument('--cavp-ckpt-path', type=str)
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fps = 4                                                     #  CAVP default FPS=4, Don't change it.
    batch_size = 40   # Don't change it.
    cavp_config_path = args.cavp_config_path            #  CAVP Config
    cavp_ckpt_path = args.cavp_ckpt_path      #  CAVP Ckpt

    # Initalize CAVP Model:
    extract_cavp = Extract_CAVP_Features(fps=fps, batch_size=batch_size, device=device, config_path=cavp_config_path, ckpt_path=cavp_ckpt_path)

    items_list_path = args.item_list_path
    item_list = open(items_list_path).readlines()

    for item in tqdm(item_list):
        item = os.path.join(args.video_path, item.strip() + '.mp4')
        if not os.path.exists(item):
            continue
        basename = os.path.basename(item)[:-4] + '.npz'
        path = os.path.join(args.output_dir, basename)
        if os.path.exists(path):
            continue

        start_second = 0              # Video start second
        truncate_second = 10.0         # Video end = start_second + truncate_second

        # Extract Video CAVP Features & New Video Path:
        feat = extract_cavp(item, start_second, truncate_second)
        np.savez(path, feat=feat)