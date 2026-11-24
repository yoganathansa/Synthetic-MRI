# Tumour-Aware Synthesis of Multi-Contrast Brain MRI

## Overview

This repository contains the implementation of a tumour-aware 3D conditional generative adversarial network (cGAN) for synthesizing multi-contrast brain MRI, including T1-CE, T2, and T2-FLAIR.

The framework investigates the effect of incorporating complementary MRI contrasts on synthesis quality and preservation of tumour-related information.

## Methodology

The proposed framework includes:

- An attention-enhanced 3D U-Net generator with a Swin Transformer bottleneck.
- Dual-scale 3D PatchGAN discriminators.
- Tumour-aware patch sampling and tumour-weighted reconstruction loss.
- Volumetric structural consistency using 3D SSIM.
- Single-input (T1-only) and multi-input synthesis configurations.

## Evaluation

The framework was evaluated using:

- **Image quality:** Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR).
- **Downstream tumour segmentation:** Dice similarity coefficient and Hausdorff distance.
- **Contrast replacement:** Evaluation of selectively replacing acquired MRI contrasts with their synthetic counterparts.
- **Clinical domain adaptation:** Fine-tuning and evaluation using an independent brain radiotherapy MRI dataset.

## Dataset

- **BraTS 2021:** Public brain tumour MRI dataset.
- **Clinical dataset:** An independent, anonymized brain radiotherapy MRI cohort.

Clinical data are not publicly distributed due to privacy and institutional restrictions.

## Intended Use

This research investigates synthetic MRI as a supplementary approach when acquired MRI contrasts are unavailable or incomplete. Synthetic images are intended to complement, not replace, acquired MRI or established clinical assessment.

## Citation

If you use this work, please cite the associated publication:

*Tumour-Aware Synthesis of Multi-Contrast Brain MRI: Image Quality and Tumour Segmentation Evaluation.*

[Publication details to be added upon acceptance/publication.]



## Contact
For questions or collaborations, please contact:
[SA Yoganathan]
[Saint John Regional Hospital]
[Saint John, New Brunswick, Canada]
[sa.yoganathan@horizonnb.ca]

