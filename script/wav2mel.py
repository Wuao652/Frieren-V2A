import os
import argparse

from glob import glob

import librosa
import numpy as np
import torchvision
import subprocess

from tqdm import tqdm

def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)

class MelSpectrogram(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels)

    def __call__(self, x):
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x, sr=self.sr, n_fft=self.nfft, fmin=self.fmin, fmax=self.fmax, power=self.spec_power
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen)) ** self.spec_power
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec
        

class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)


class Add(object):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val


class Subtract(Add):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val


class Multiply(object):
    def __init__(self, val, inverse=False):
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val


class Divide(Multiply):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10(object):
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10 ** x
        else:
            return np.log10(x)


class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)


class TrimSpec(object):
    def __init__(self, max_len, inverse=False):
        self.max_len = max_len
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x[:, :self.max_len]


class MaxNorm(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
        self.eps = 1e-10

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x / (x.max() + self.eps)


def get_spectrogram(audio_path, length, mel_tran, sr=16000):
    wav, sr_new = librosa.load(audio_path, sr=sr)
    wav = wav.reshape(-1)

    y = np.zeros(length)
    if wav.shape[0] < length:
        y[:len(wav)] = wav
    else:
        y = wav[:length]
    
    y = y[ : length - 1]      
    mel_spec = mel_tran(y)
    return y, mel_spec


def tran_mel(gen_audio_root,chunk_time=8.2,sr=16000,nmels=128):
    mel_tran = torchvision.transforms.Compose([
        MelSpectrogram(sr=sr, nfft=1024, fmin=125, fmax=7600, nmels=nmels, hoplen=256, spec_power=1),
        LowerThresh(1e-5),
        Log10(),
        Multiply(20),
        Subtract(20),
        Add(100),
        Divide(100),
        Clip(0, 1.0),
    ])
    if sr == 16000:
        str_sr = '16k'
    else:
        str_sr = str(sr)

    tran_spec_root = os.path.join(os.path.dirname(gen_audio_root), os.path.basename(gen_audio_root) + f"_mel_{nmels}_{chunk_time}") 

    gen_audio_paths = sorted(glob(f'{gen_audio_root}/*.wav'))
    print(len(gen_audio_paths))
    
    length = int(sr * chunk_time) 

    os.makedirs(tran_spec_root, exist_ok=True)
    for input_wav_path in tqdm(gen_audio_paths):
        if sr == 22050:
            class_num = get_class(os.path.basename(input_wav_path))
            os.makedirs(os.path.join(tran_spec_root, f'cls_{class_num}'), exist_ok=True)
            mel_path = os.path.join(tran_spec_root, f'cls_{class_num}', os.path.basename(input_wav_path)[:-4]+'.npy')
        elif sr == 16000:
            os.makedirs(tran_spec_root, exist_ok=True)
            mel_path = os.path.join(tran_spec_root, os.path.basename(input_wav_path)[:-4]+'.npy')
        if (os.path.exists(mel_path)):
            continue
        _, mel_spec = get_spectrogram(input_wav_path, length, mel_tran, sr)
        np.save(mel_path, mel_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        help='audio dir',
    )

    args = parser.parse_args()

    chunk_time = 8.2
    sr = 16000
    nmels = 128

    tran_mel(args.dir, chunk_time, sr, nmels)
