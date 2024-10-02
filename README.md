# FMint: Bridging human designed and data pretrained models for differential equation foundation model

The repository contains the code for the following paper:
- [FMint: Bridging Human Designed and Data Pretrained Models for Differential Equation Foundation Model](https://arxiv.org/abs/2404.14688).

## Environment Setup

The YAML file `environment.yml` contains the environment setup for the code. To create the environment, run the following command:
```
conda env create -f environment.yml
```
which will create an environment named `icon`. You may need to update the `prefix` field in the YAML file to specify the location of the environment.

## Data Preparation

The code for function data generation is located in the `data_generation/` folder. To generate pre-training data, run
```
bash datagen_neuralvec.sh
```

## Training

To pretrain (fine-tune) the model, run `run.sh`.

## Analysis and Visualization

The analysis code and run commands are located in the `analysis/` folder.

## Citation

If find this code repository helpful, please consider citing the following paper:

```
@article{song2024fmint,
  title={Fmint: Bridging human designed and data pretrained models for differential equation foundation model},
  author={Song, Zezheng and Yuan, Jiaxin and Yang, Haizhao},
  journal={arXiv preprint arXiv:2404.14688},
  year={2024}
}
```

### Acknowledgement
This repository is adapted from [In-Context Operator Networks (ICON)](https://github.com/LiuYangMage/in-context-operator-networks).

