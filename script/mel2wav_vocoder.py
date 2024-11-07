import os
import sys
import glob
import argparse
import numpy as np
import soundfile as sf

from tqdm import tqdm

sys.path.append(os.path.join("/".join(os.getcwd().split("/")[:-1]), "Frieren"))
from vocoder.bigvgan.models import VocoderBigVGAN
    
success_list = []
err_list = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root',
        type=str,
    )
    parser.add_argument(
        '--vocoder-path',
        type=str,
    )
    
    args = parser.parse_args()

    vocoder = VocoderBigVGAN(args.vocoder_path)

    root = args.root
    target_dir = os.path.join(os.path.dirname(root), os.path.basename(root) + 'gen_wav_16k_80')
    os.makedirs(target_dir, exist_ok=True)
    
    for mel_file in tqdm(sorted(glob.glob(f'{root}/*.npy'))):
        wav_file = os.path.join(target_dir, mel_file.split('/')[-1].replace('.npy', '.wav'))
        if os.path.exists(wav_file):
            continue
        mel_spec = np.load(mel_file)
        sample = vocoder.vocode(mel_spec)
        sf.write(wav_file, sample, 16000)
