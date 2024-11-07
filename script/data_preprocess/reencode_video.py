import os
import argparse
import torch
from tqdm import tqdm

import subprocess
from pathlib import Path


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: int, start_second, truncate_second) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (int): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    
    new_path = os.path.join(tmp_path, f'{Path(video_path).stem}.mp4')
    cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
    cmd += f'-y -ss {start_second} -t {truncate_second} -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
    
    subprocess.call(cmd.split())
    
    return new_path


class VideoReencoder(torch.nn.Module):

    def __init__(self, fps=4, batch_size=2, device=None):
        super(VideoReencoder, self).__init__()
        self.fps = fps
        self.batch_size = batch_size
        self.device = device
    
    
    @torch.no_grad()
    def forward(self, video_path, start_second=None, truncate_second=None, tmp_path="./tmp_folder"):
        reencode_video_with_diff_fps(video_path, tmp_path, self.fps, start_second, truncate_second)
        

if __name__ == '__main__':
    device = torch.device('cuda')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--item-list-path', type=str)
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    fps = 4                                                     #  CAVP default FPS=4, Don't change it.
    batch_size = 40   # Don't change it.

    # Initalize CAVP Model:
    video_reencoder = VideoReencoder(fps=fps, batch_size=batch_size, device=device)

    items_list_path = args.item_list_path
    item_list = open(items_list_path).readlines()

    for item in tqdm(item_list):
        video_path = os.path.join(args.input_dir, item.strip() + '.mp4')
        if not os.path.exists(video_path):
            continue

        start_second = 0              # Video start second
        truncate_second = 10.0         # Video end = start_second + truncate_second

        video_reencoder(video_path, start_second, truncate_second, tmp_path=args.output_dir)
