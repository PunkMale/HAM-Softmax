<div align="center">

# HAM-Softmax: Hyperbolic Additive Margin Softmax with Hierarchical Information for Speaker Verification

[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-11463316-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/abstract/document/11463316)
[![arXiv](https://img.shields.io/badge/arXiv-2601.19709-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2601.19709)

Official implementation of **Hyperbolic Additive Margin Softmax with Hierarchical Information for Speaker Verification**, accepted by **ICASSP 2026**.

</div>


## Data Preparation

Prepare VoxCeleb or CN-Celeb style audio directories, then replace the dataset roots in [utils.py](utils.py) with your local paths:

```python
vox1_dev_path = '/path/to/voxceleb1/dev/wav/'
vox2_dev_path = '/path/to/voxceleb2/dev/aac/'
cn_dev_path = '/path/to/cnceleb'
eval_path = '/path/to/voxceleb1'
```

The repository uses list files under [data](data):

```text
data/
  v1_clean.txt
  v2_clean.txt
  cn_clean.txt
  vox_O.txt
  vox_E.txt
  vox_H.txt
  vox_EH_list.txt
  CN.Eval_list.txt
```

Training list format:

```text
speaker_id relative/path/to/audio.wav
```

VoxCeleb-O/E/H evaluation list format:

```text
label enroll_utterance test_utterance
```

For CN-Celeb evaluation, replace the corresponding paths in [utils.py](utils.py):

```python
eval_list1 = '/path/to/cn_veri_test.txt'
eval_list2 = '/path/to/cn_veri_test.txt'
eval_path = '/path/to/cnceleb/cn_1/eval'
```

For augmentation, replace the MUSAN and RIR paths in [datasets.py](datasets.py):

```python
self.musan_path = "/path/to/musan"
self.rir_path = "/path/to/RIRS_NOISES/simulated_rirs"
```

## Training

General training command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset v2 \
  --lambda_1 0 \
  --lambda_2 1 \
  --h_C 3 \
  --h_s 30 \
  --h_m 0.2 \
  --batch_size 256 \
  --augment
```

Important arguments:

| Argument | Description |
| --- | --- |
| `--dataset` | Dataset key: `v1`, `v2`, or `cn`. |
| `--loss_type` | Euclidean auxiliary loss: `ce`, `ces`, `am`, `aam`, or `ram`. |
| `--lambda_1` | Weight of the Euclidean loss branch. |
| `--lambda_2` | Weight of the hyperbolic loss branch. |
| `--h_C` | Hyperbolic curvature parameter. |
| `--h_m` | Hyperbolic additive margin. Use `0` for H-Softmax. |
| `--h_s` | Hyperbolic scale factor. |
| `--augment` | Enable waveform augmentation. |

## Example Experiments

The provided [run.sh](run.sh) contains three example VoxCeleb1 runs:

```bash
./run.sh
```

They correspond to:

| Experiment | Key setting |
| --- | --- |
| H-Softmax | `h_C=5`, `h_m=0`, `h_s=30`, `lambda_1=0`, `lambda_2=1` |
| HAM-Softmax | `h_C=3`, `h_m=0.2`, `h_s=30`, `lambda_1=0`, `lambda_2=1` |
| E.H. AM-Softmax | `loss_type=ram`, `h_C=3`, `h_m=0.2`, `h_s=30`, `lambda_1=0.3`, `lambda_2=0.7` |

## Evaluation and Outputs

During training, the script evaluates periodically according to the dataset setting in [utils.py](utils.py). Results and checkpoints are saved under:

```text
exps/<experiment_name>/
  train.log
  result_<experiment_name>.csv
  Vox-EH.csv
  model/
```

When training on VoxCeleb2 (`--dataset v2`), the final model is additionally evaluated on VoxCeleb-E and VoxCeleb-H with the fast evaluation dataset. The results are saved to `Vox-EH.csv`.


## Citation

If this repository is helpful for your research, please cite:

```bibtex
@INPROCEEDINGS{11463316,
  author={Fang, Zhihua and He, Liang},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Hyperbolic Additive Margin Softmax with Hierarchical Information for Speaker Verification},
  year={2026},
  pages={19017-19021},
  doi={10.1109/ICASSP55912.2026.11463316}
}
```
