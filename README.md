# Synthetic-MRI
Dual-scale 2.5D conditional GAN for multi-contrast brain MRI synthesis. Generates T1-CE, T2, and FLAIR images from T1 MRI, reducing scan time, contrast agent use, and motion artifacts in radiotherapy imaging.

## Overview
This project implements a deep learning framework for generating synthetic multi-contrast brain MRI from a single T1-weighted image. The model synthesizes the following MRI contrasts:

- T1 contrast-enhanced (T1-CE)
- T2-weighted MRI
- T2-FLAIR MRI
- 
The framework is based on a conditional Generative Adversarial Network (cGAN) with dual-scale discriminators and a 2.5D input strategy that incorporates neighbouring slices to improve volumetric consistency.

The primary goal is to reduce MRI scan time, decrease reliance on gadolinium-based contrast agents, and mitigate motion artifacts in brain radiotherapy workflows.

---

## Key Features

- Dual-scale GAN architecture capturing local and global anatomical features
- 2.5D slice-based synthesis using neighbouring slice context
- Hybrid loss function combining:
  - Adversarial loss
  - Feature matching loss
  - Perceptual loss
  - 3D volumetric consistency loss
- Evaluation using both image similarity metrics and tumour segmentation performance
- Designed for radiotherapy planning applications

---

## Dataset

### Public Dataset
BraTS 2021 dataset

Training: 1000 cases  
Validation: 100 cases  
Testing: 100 cases  

Each case includes:
- T1-weighted MRI
- T1 contrast-enhanced MRI
- T2-weighted MRI
- T2-FLAIR MRI
- Tumour segmentation masks

### Clinical Dataset
Independent brain radiotherapy dataset

Training (fine-tuning): 20 cases  
Testing: 10 cases  

MRI scans include:
- T1
- T1-CE
- T2
- T2-FLAIR

All images were preprocessed with:
- Denoising
- Bias field correction
- Rigid registration
- Resampling to 1×1×1 mm resolution

---

## Model Architecture

### Generator
Encoder–decoder architecture inspired by pix2pixHD with attention modules.

Input:
- T1 slice with neighbouring slices (2.5D input)

Output:
- Synthetic MRI contrast (T1-CE / T2 / T2-FLAIR)

### Discriminators
Two discriminators operate at different spatial scales:

1. Full-resolution discriminator
2. Half-resolution discriminator (0.5×)

These ensure both fine texture realism and global anatomical consistency.

---

## Training Strategy

The generator is trained using a hybrid loss function:

L_G = λ1 AL_2D + λ2 FML_2D + λ3 PL_2D + λ4 MAE_2.5D + λ5 AL_2.5D

Where:

AL = adversarial loss  
FML = feature matching loss  
PL = perceptual loss  
MAE = mean absolute error on reconstructed volume  

Loss weights:

λ1 = 1  
λ2 = 5  
λ3 = 5  
λ4 = 5  
λ5 = 5  

---

## Evaluation Metrics

Synthetic MRI quality was evaluated using:

- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)

Clinical utility was assessed using tumour segmentation metrics:

- Dice Similarity Coefficient
- Hausdorff Distance

---

## Clinical Validation

The synthetic MRI images were tested for tumour segmentation performance using a 3D segmentation model.

Experiments included:

1. Using all original MRI contrasts
2. Replacing individual contrasts with synthetic images
3. Using fully synthetic MRI contrasts

Results showed that synthetic MRI can effectively supplement missing contrasts, particularly for whole tumour segmentation.


---

## Applications

- Brain radiotherapy planning
- MRI protocol acceleration
- Contrast-agent reduction
- Motion artifact mitigation
- Data augmentation for medical imaging models

---

## Citation

If you use this work, please cite:

Multi-Contrast Brain MRI Synthesis for Radiotherapy Using a Dual-Scale Generative Adversarial Network with Volumetric Context

---



## Contact
For questions or collaborations, please contact:
[SA Yoganathan]
[Saint John Regional Hospital]
[Saint John, New Brunswick, Canada]
[sa.yoganathan@horizonnb.ca]

