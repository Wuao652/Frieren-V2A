# Frieren: Efficient Video-to-Audio Generation Network with Rectified Flow Matching
#### Yongqi Wang*, Wenxiang Guo*, Rongjie Huang, Jiawei Huang, Zehan Wang, Fuming You, Ruiqi Li, Zhou Zhao
#### Zhejiang University

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2406.00320) [![Demo](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue.svg?logo=Github)](https://frieren-v2a.github.io/)  

This is the PyTorch implementation of Frieren (NeurIPS'24), a transformer-based non-autoregressive video-to-audio generation model with rectified flow matching. 

## Pretrained checkpoints and 25-step generated results

Here, we provide three major version of our checkpoints on VGGSound with CAVP as the visual feature:

* [Frieren (no reflow)](https://drive.google.com/drive/folders/1pzsVP5kXkTn5xzHWvZ4lurcinN4_Bg__?usp=drive_link)
* [Frieren (reflow)](https://drive.google.com/drive/folders/1pzMLB66DggC3IaVEREnjrFrzwdpR-A4y?usp=drive_link)
* [Frieren (reflow + distillation)](https://drive.google.com/drive/folders/1q-JjBkEdBpBvPhTHl-bdKP6BqYGWiewc?usp=drive_link)

The checkpoints of VAE and vocoder are provided here:

* [VAE](https://drive.google.com/drive/folders/11rtgLedRjT4lckud0BZQ083APIPnLGoe?usp=drive_link)
* [vocoder](https://drive.google.com/drive/folders/11tg7VagVNvucNtQ8y0R4UJLIhEeR76oj?usp=drive_link)

For CAVP, please download the checkpoint from the [Diff-Foley repo](https://github.com/luosiallen/Diff-Foley).

For ease of comparison, we have also uploaded the 25-step sampling results on the of the no-reflow version on VGGSound test set to [this URL](https://huggingface.co/datasets/Cyanbox/Frieren-V2A-results/blob/main/frieren_noreflow_25.zip). These are the generated results we used for calculating metrics.  

## Environment

See `requirements.txt`

## Data preprocessing

### Data splits

The item names of training and test sets of VGGSound we use are in `data/Train.txt` and `data/Test.txt`.

### Visual feature extraction

For CAVP, we follow the method of extracting in Diff-Foley.

* step 1: Downsample videos to 4 FPS. Run the following command in `${FRIEREN_ROOT_DIR}`
    ```shell
    python script/data_preprocess/reencode_video.py \
     --input-dir ${VIDEO_MP4_DIR} \
     --output-dir ${VIDEO_4FPS_DIR} \
     --item-list-path ${ITEM_LIST_FPATH}
    ```
  where the format of `${ITEM_LIST_FPATH}` is the same as `data/Train.txt`.

* step 2: Download CAVP checkpoint from the [Diff-Foley repo](https://github.com/luosiallen/Diff-Foley).

* step 3: Extract CAVP features. Run the following command in `${FRIEREN_ROOT_DIR}`.
    ```shell
    export PYTHONPATH=$PYTHONPATH:${FRIEREN_ROOT_DIR}/Frieren
    python script/data_preprocess/extract_cavp_feature.py \
     --item-list-path ${ITEM_LIST_FPATH} \
     --cavp-config-path ${FRIEREN_ROOT_DIR}/Frieren/configs/cavp.yaml \
     --cavp-ckpt-path ${CAVP_CKPT_PATH} \
     --video-path ${VIDEO_4FPS_DIR} \
     --output-dir ${CAVP_OUTPUT_DIR}
    ```

  Adjust the scripts for your own format of data.

For MAViL, we utilize the model and checkpoint provided in [av-superb](https://github.com/roger-tseng/av-superb). As MAViL take 4-second 2-fps (8 frames) video clips, we can devide each 10-second 4-fps (40 frames) video clip to 5 sub-clips (or 8-second 4-fps video clip to 4 sub-clips) for extraction, and concat the features in the temporal dimension.

### Mel-spectrogram extraction
Run the following command in `${FRIEREN_ROOT_DIR}` for extracting mel-spectrogram for training.

```shell
export PYTHONPATH=$PYTHONPATH:${FRIEREN_ROOT_DIR}/script/ 
python script/data_preprocess/mel_spec.py --manifest-path ${ITEM_LIST_FPATH} \
 --input-dir ${AUDIO_DIR} \
 --output-dir ${MEL_DIR}
```

### Filter corrupted data

Some of the training or test items may cause corrupted CAVP feature or mel-spectrogram, you may filter them out from the manifest before training or inference.

## Inference

### Mel generation

1. Change `model.params.first_stage_config.params.ckpt_path` in `Frieren/configs/ldm_training/v2a_cfm_cfg_infer.yaml` to your local path of the VAE checkpoint.
2. Switch to directory `script` and run:
    ```shell
    python generate_mel_cfm_cfg_scale.py \
      -m ${CFM_CKPT_PATH} \
      -c ${FRIEREN_ROOT_DIR}/Frieren/configs/ldm_training/v2a_cfm_cfg_infer.yaml \
      -g 0 \
      --cavp_feature ${CAVP_FEATURE_DIR}
      --out-dir ${OUT_DIR} \
      -s euler \  # euler, rk4, dopri5
      -t 26 \  # should be sampling step + 1, here 26 means 25 sampling step
      --scale 4.5 \  # CFG scale
      --sample-num 10  # number of samples
    ```

    The sampled spectrogram will be stored in a subdirectory under `${OUT_DIR}`.

### Wave generation

Switch to directory `script` and run:

```shell
python mel2wav_vocoder.py --root ${MEL_DIR} --vocoder-path ${VOCODER_CKPT_CONFIG_DIR}
```

The generated audio will be stored in a directory at the same level as the spectrogram directory, with the suffix `gen_wav_16k_80`.

## Evaluation

For alignment accuracy, we convert the audio generated by the vocoder into a spectrogram and perform calculations. 

1. Switch to directory `script` and run:
    ```shell
    python wav2mel.py --dir ${AUDIO_DIR}
    ```
    The spectrogram will be stored in a directory at the same level as `${AUDIO_DIR}`, with the suffix `_mel_128_8.2`.

2. Refer to [alignment classifier in Diff-Foley](https://github.com/luosiallen/Diff-Foley/tree/main/evaluation#align-acc) for evaluation.

For other metrics, please refer to [audioldm-eval](https://github.com/haoheliu/audioldm_eval).

> **Note:** When calculating metrics with references (such as FAD), in order to ensure that the number of reference audio and evaluated audio is consistent, we duplicate each reference (gt) audio items for 10 times (since we generate 10 samples for each data item) and align their names with the generated samples. Not doing so may lead to differences in FAD values with our results, but will (probably) not alter the performance ranking between different models.

## Training

### Training from scratch

1. Prepare training data. Let `${DATA_ROOT}/vggsound` be the dataset root.
   * Put `data/Train.txt` and `data/Test.txt` in `${DATA_ROOT}/vggsound/split_txt`
   * Put extracted spectrograms in `${DATA_ROOT}/vggsound/mel`
   * Put cavp features in `${DATA_ROOT}/vggsound/cavp`

2. Prepare training configuration file. The base configuration file is `${FRIEREN_ROOT_DIR}/Frieren/configs/ldm_training/v2a_cfm_train_load_novidpos_concat_2card.yaml`.
    * Replace the paths of submodule checkpoints and dataset to your local paths.
    * [OPTIONAL] Use the denoiser weight make-an-audio 2 to initialize the vector field estimator using the `model.params.model_ckpt_path` parameter.

3. Run the following command in directory `Frieren`.

    ```shell
    python main.py --base ${FRIEREN_ROOT_DIR}/Frieren/configs/ldm_training/v2a_cfm_train_load_novidpos_concat_2card.yaml -t --gpus 0,1 --stage 2 --epoch 250 --scale_lr False -l ${OUTPUT_DIR}
    ```

    To recover training from a log diretory, replace the parameter `-l ${OUTPUT_DIR}` to `-r ${OUTPUT_DIR}\${EXP_SUBDIR}`

### Reflow-data generation

Run the following command in directory `script`.

```shell
python generate_mel_noise_cfm_cfg_scale.py \
 -m ${CFM_CKPT_PATH} \
 -c ${FRIEREN_ROOT_DIR}/Frieren/configs/ldm_training/v2a_cfm_cfg_infer.yaml \
 -g 0 \
 --cavp-feature-dir ${CAVP_FEATURE_DIR} \
 --cavp-feature-list ${FRIEREN_ROOT_DIR}/data/Train.txt \
 --out-dir ${REFLOW_DATA_DIR} \
 -s euler -t 26 --scale 4.5 
```

### Reflow

1. Prepare training configuration file. The base configuration file is `${FRIEREN_ROOT_DIR}/Frieren/configs/ldm_training/v2a_cfm_reflow_vcfg_mix_train_load_novidpos_concat_2card.yaml`.
    * Replace the paths of submodule checkpoints and dataset to your local paths. The additional `data.params.train.params.dataset1.reflow_dir` and `data.params.validation.params.dataset1.reflow_dir` should be the directory of noise and sampled data generated in previous reflow-data generation step.
    * Replace `model.params.model_ckpt_path` to the path of the no-reflow checkpoint.

2. Run the following command in directory `Frieren`.

    ```shell
    python main.py --base ${FRIEREN_ROOT_DIR}/v2a_cfm_reflow_vcfg_mix_train_load_novidpos_concat_2card.yaml -t --gpus 0,1 --stage 2 --epoch 250  --scale_lr False -l ${OUTPUT_DIR}
    ```

### Distillation

1. Prepare training configuration file. The base configuration file is `${FRIEREN_ROOT_DIR}/Frieren/configs/ldm_training/v2a_cfm_distill_vcfg_mix_train_load_novidpos_concat_2card.yaml`.
    * Replace the paths of submodule checkpoints and dataset to your local paths. The additional `data.params.train.params.dataset1.reflow_dir` and `data.params.validation.params.dataset1.reflow_dir` should be the directory of noise and sampled data generated in previous reflow-data generation step.
    * Replace `model.params.model_ckpt_path` to the path of the reflow checkpoint.

2. Run the following command in directory `Frieren`.

    ```shell
    python main.py --base ${FRIEREN_ROOT_DIR}/v2a_cfm_distill_vcfg_mix_train_load_novidpos_concat_2card.yaml -t --gpus 0,1 --stage 2 --epoch 250  --scale_lr False -l ${OUTPUT_DIR}
    ```
    
## Acknowledgements
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.

* Diff-Foley: https://github.com/luosiallen/Diff-Foley
* Make-an-Audio 2: https://github.com/bytedance/Make-An-Audio-2

## Citations 
If you find this code useful in your research, please cite our work:
```bib
@article{wang2024frieren,
  title={Frieren: Efficient Video-to-Audio Generation with Rectified Flow Matching},
  author={Wang, Yongqi and Guo, Wenxiang and Huang, Rongjie and Huang, Jiawei and Wang, Zehan and You, Fuming and Li, Ruiqi and Zhao, Zhou},
  journal={arXiv preprint arXiv:2406.00320},
  year={2024}
}

@article{huang2023make,
  title={Make-an-audio: Text-to-audio generation with prompt-enhanced diffusion models},
  author={Huang, Rongjie and Huang, Jiawei and Yang, Dongchao and Ren, Yi and Liu, Luping and Li, Mingze and Ye, Zhenhui and Liu, Jinglin and Yin, Xiang and Zhao, Zhou},
  journal={arXiv preprint arXiv:2301.12661},
  year={2023}
}
```
