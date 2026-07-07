# ADM-JRM

Official implementation of our paper on adaptive diffusion model based joint reconstruction and motion compensation.

## Installation

Create and activate your Python environment, then install `torch-radon` with conda:

```bash
conda install conda-forge::carterbox-torch-radon
```

Install the remaining Python requirements:

```bash
pip install -r requirements.txt
```

## Pretrained Weights

Download the pretrained W3DM weights and place them at the path expected by the YAML config:

```bash
mkdir -p weights
wget -O weights/model_state_dict.pth "https://huggingface.co/antoinedepaepe/adm-jrm-w3dm/resolve/main/model_state_dict.pth?download=true"
```

The default config expects:

```yaml
adm_jrm:
  model_path: "./weights/model_state_dict.pth"
```

## Configuration

Before running the code, set the experiment parameters in:

```bash
config/adm_jrm.yaml
```

This file controls the device, data location, acquisition geometry, motion parameters, noise level, diffusion schedule, optimizer settings, and output paths. We refer to the paper for the detailed parameter choices.

## Generate Motion-Affected Measurements

First generate simulated motion-corrupted cone-beam measurements:

```bash
python generate_data.py
```

## Run JRM-ADM Reconstruction

Then run JRM-ADM reconstruction:

```bash
python run_jrm_adm.py
```


## Citation

If you use this code, please cite our paper:

```bibtex
@article{de2025adaptive,
  title={Adaptive Diffusion Models for Sparse-View Motion-Corrected Head Cone-Beam CT},
  author={De Paepe, Antoine and Bousse, Alexandre and Phung-Ngoc, Cl{\'e}mentine and Mellak, Youness and Visvikis, Dimitris},
  journal={IEEE Transactions on Radiation and Plasma Medical Sciences},
  year={2025},
  publisher={IEEE}
}
```
