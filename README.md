# HAFTrack: Official Code for "HAFTrack:Heterogeneous Asymmetric Fusion in RGBT Tracking"
**Author**: {DurianLatte913} | {University of Electronic Science and Technology of China}

**Paper**: {HAFTrack}

**GitHub**: https://github.com/DurianLatte913/HAFTrack-main

## Introduction
![Image Caption](imgs/overall.png)
This repository contains the official implementation of the RGBT (RGB-Thermal) object tracking algorithm proposed in our paper "HAFTrack:Heterogeneous Asymmetric Fusion in RGBT Tracking". Our method proposed a novel multimodal tracking framework, termed HAFTrack, which is specifically designed to accommodate the intrinsic characteristics of RGB and  TIR modalities as well as the fundamental requirements of RGBT tracking, thereby fully exploiting the high-value information provided by both modalities.


## Device
We trained and tested our model on four NVIDIA RTX 4090 GPUs.


## Environment Configuration
### Prerequisites
Our code has been tested in the following environment:
* Python >= 3.8
* PyTorch >= 1.10.1
* CUDA >= 11.3

### Install Dependencies
First, clone this repository and navigate to the project directory:
```bash
git clone https://github.com/DurianLatte913/HAFTrack-main.git
cd HAFTrack-main
```
Install the required packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```
