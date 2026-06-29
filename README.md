# ISRO Hackathon – Infrared Satellite Image Enhancement and Colorization

## Overview

This project presents an end-to-end deep learning framework for enhancing and colorizing infrared (IR) satellite imagery. The solution is designed to improve the visual quality and interpretability of monochrome IR images by restoring structural details and translating them into realistic RGB representations.

Infrared satellite images are widely used for night-time observation, disaster monitoring, and remote sensing under challenging weather conditions. However, these images often suffer from low contrast, limited texture information, and lack of natural color cues, making object identification difficult for both human analysts and computer vision systems.

This project addresses these challenges through a multi-stage pipeline that combines image enhancement, attention-based deep learning, adversarial training, and perceptual evaluation.

## Key Features

* Automated IR dataset preparation and patch generation
* Infrared image enhancement using:

  * CLAHE (Contrast Limited Adaptive Histogram Equalization)
  * Gamma Correction
  * Bilateral Filtering
  * Unsharp Masking
* Residual U-Net Generator with:

  * CBAM (Convolutional Block Attention Module)
  * Skip Attention Gates
  * Residual Learning
* Multi-Scale GAN Training Framework
* Advanced loss functions:

  * Adversarial Loss
  * L1 Reconstruction Loss
  * Perceptual Loss (VGG19)
  * SSIM Loss
  * Edge Preservation Loss
* Mixed Precision Training (CUDA)
* Automatic checkpointing and TensorBoard logging
* Comprehensive evaluation using:

  * PSNR
  * SSIM
  * LPIPS

## Project Pipeline

IR Satellite Images
→ Dataset Preparation
→ Image Enhancement
→ Residual U-Net + Attention Generator
→ RGB Image Generation
→ Multi-Scale Adversarial Training
→ Quantitative Evaluation

## Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* TorchVision
* TensorBoard
* CUDA

## Applications

* Satellite Remote Sensing
* Disaster Management
* Night-Time Monitoring
* Environmental Analysis
* Infrastructure and Road Detection
* Automated Object Interpretation
* Geospatial Intelligence

## Results

The framework aims to generate visually realistic RGB images from infrared satellite data while preserving critical spatial structures such as roads, buildings, vegetation, and vehicles. Evaluation is performed using PSNR, SSIM, and LPIPS metrics to measure reconstruction quality and perceptual similarity.

## Developed For

ISRO Hackathon – Satellite Remote Sensing Challenge

Enhancing low-visibility infrared satellite imagery through deep learning-based image enhancement and colorization to improve situational awareness and automated interpretation.

