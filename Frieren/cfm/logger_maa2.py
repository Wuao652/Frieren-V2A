import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchvision
import pytorch_lightning as pl
import numpy as np
import soundfile as sf
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import librosa
import os
import importlib
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    print(config['target'])
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


class SoundLogger_concat_fullset(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False, vocoder_cfg=None,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.fps = fps
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        print('sr: ', self.sr)
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond
        print("Guidance Scale: ", self.guidance_scale)
        print("Uncond cond: ", self.uncond_cond)

        self.vocoder = instantiate_from_config(vocoder_cfg)

    
    def inverse_op(self, spec):
        wav = self.vocoder.vocode(spec)

        return wav
        

    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        gt_sound_list = log_dict['inputs_spec'].detach().cpu().numpy()
        rec_sound_list = log_dict['reconstruction_spec'].detach().cpu().numpy()
        diff_sample_list = log_dict['samples'].detach().cpu().numpy()

        os.makedirs(root,exist_ok=True)
        
        mix_info_dict = log_dict["mix_info_dict"]

        for i in range(len(gt_sound_list)):
            
            if mix_info_dict['audio_name2'][i] == "":
                video_path_list = mix_info_dict['video_path1']
                video_time_list = mix_info_dict['video_time1'] 
                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = self.inverse_op(gt_sound_list[i])
                rec_sound = self.inverse_op(rec_sound_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sample = self.inverse_op(diff_sample_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
                
                try:
                    video = self.extract_concat_frame_video(video_path_list[i], video_time_list[i], out_folder=sample_folder)
                except Exception as e:
                    print(e)
                    pass

                with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list[i]) + "    " + str(video_time_list[i])
                    f.writelines(txt)
            
            else:
                video_path_list1, video_path_list2 = mix_info_dict['video_path1'], mix_info_dict['video_path2']
                video_time_list1, video_time_list2 = mix_info_dict['video_time1'], mix_info_dict['video_time2'] 

                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = self.inverse_op(gt_sound_list[i])
                rec_sound = self.inverse_op(rec_sound_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_concat_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_concat_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sample = self.inverse_op(diff_sample_list[i])
                sf.write(os.path.join(sample_folder, "sample_{}_diff_concat_sample_clamp.wav".format(i)), sample, self.sr) 
                
                try:
                    video = self.extract_concat_frame_video(video_path_list1[i], video_time_list1[i], video_path_list2[i], video_time_list2[i], out_folder=sample_folder)
                except:
                    pass

                with open(os.path.join(sample_folder, "video_path_cat.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list1[i]) + "    " + str(video_time_list1[i]) + '\n' + "Video 2:" + '  ' + str(video_path_list2[i]) + "    " + str(video_time_list2[i])
                    f.writelines(txt)
        

    def extract_concat_frame_video(self, video_path1, video_time1, video_path2=None, video_time2=None, out_folder=None):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame1, end_frame1 = int(video_time1.split('_')[0]), int(video_time1.split('_')[1])
        start_time1, end_time1 = start_frame1 / self.fps, end_frame1 / self.fps
        src_path1 = video_path1
        out_path = os.path.join(out_folder, "origin.mp4")

        video1 = VideoFileClip(src_path1).subclip(start_time1, end_time1)

        if video_path2 is not None:
            start_frame2, end_frame2 = int(video_time2.split('_')[0]), int(video_time2.split('_')[1])
            start_time2, end_time2 = start_frame2 / self.fps, end_frame2 / self.fps
            src_path2 = video_path2
            out_path = os.path.join(out_folder, "origin_cat.mp4") 
            video2 = VideoFileClip(src_path2).subclip(start_time2, end_time2)
            finalclip = concatenate_videoclips([video1, video2], method="compose")

            finalclip.write_videofile(out_path)
        else:
            video1.write_videofile(out_path)

        return True

        
    @rank_zero_only
    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch,N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx)

        if is_train:
            pl_module.train()


    def check_frequency(self, check_idx, split):
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass


    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

