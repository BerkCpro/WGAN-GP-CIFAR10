# WGAN-GP Image Generation on CIFAR-10

This repository contains a PyTorch implementation of a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) trained from scratch to generate images based on the CIFAR-10 dataset.

##  Project Overview
The goal of this project is to build a solid and stable generative model. By implementing the Wasserstein loss and a Gradient Penalty, this architecture successfully mitigates common GAN training issues such as mode collapse and vanishing gradients.

## Key Features
* **Custom Architecture:** Deep convolutional Generator and Critic networks designed from scratch.
* **Training Stability:** Replaced standard batch normalization in the Critic with **Instance Normalization** to maintain the validity of the gradient penalty.
* **Experiment Tracking:** Fully integrated with [Weights & Biases (WandB)](https://wandb.ai/) for real-time logging of Generator/Critic loss metrics and generated image grids.

##  Usage
*1. Clone the repository:*
   ```bash
   git clone [https://github.com/YOUR_USERNAME/CIFAR10-WGAN.git](https://github.com/YOUR_USERNAME/CIFAR10-WGAN.git)
   cd CIFAR10-WGAN
   ```
*2. Log in to your Weights & Biases account (required for logging):*
```bash
wandb login
```
*3. Run the training script:*
```bash
python wgan_cifar10.py
```

## Monitoring
*During training, the script will automatically log:*

**Disc_loss:** The Critic's loss (Wasserstein distance + Gradient Penalty).

**Gen_loss:** The Generator's loss.

**Images:** A normalized grid of fake images generated from fixed noise to track visual progress over epochs.

   
