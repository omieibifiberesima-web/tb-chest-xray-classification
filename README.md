# Tuberculosis Detection from Chest X-rays Using a CNN

A convolutional neural network (CNN) trained to classify chest X-rays as 
Normal or Tuberculosis (TB), with a focus on examining what the model gets 
wrong and what limits the results, not just overall accuracy.

This project was motivated by my clinical internship in Nigeria, where I 
saw many TB cases and where access to radiology expertise can be limited.

**This is an educational project, not a clinically validated diagnostic tool.**

## Data

Tuberculosis (TB) Chest X-ray Database (Kaggle): 4,200 images, 3,500 Normal 
and 700 TB. The images are not included in this repository. The full 
database lists 3,500 TB images, but 2,800 require a data-sharing agreement 
with NIAID, so the public version used here contains 700.

## Method

- CNN built from scratch in TensorFlow/Keras: three convolution and 
  max-pooling blocks (32, 64, 64 filters), one dense layer, sigmoid output 
  (about 2.8 million parameters)
- Images resized to 224 x 224, pixel values scaled to 0-1
- 80/20 image-level train/validation split (3,360 / 840 images)
- Adam optimizer, binary cross-entropy, 10 epochs on a Colab T4 GPU

## Results (validation set, first run)

- Accuracy: 98.6%
- TB recall: 94.2% (131 of 139 TB images detected, 8 missed)
- TB precision: 97.0% (4 false alarms among 701 Normal images)

A second run of identical code gave different results (6 TB images missed, 
14 false alarms), because neural-network training includes random elements 
that I did not fix with a seed. Both runs are reported in the notebook.

## What I learned from the errors

I reviewed the 8 missed TB images myself. Several looked clearly abnormal 
even to me, a physician without radiology training, which shows that strong 
aggregate metrics do not guarantee the model fails only on subtle cases. 
These are informal observations that have not been confirmed by a radiologist.

## Limitations

- Single dataset, with no external validation
- TB and Normal images came largely from different source collections 
  (e.g., most Normal images from the RSNA Pneumonia Detection Challenge), so 
  the model may partly learn differences between collections instead of TB
- No separate test set, and an image-level split with no patient IDs 
- Only a few errors to analyze, and results varied between runs

## Files

- `TB_Xray_CNN.ipynb`: full notebook with code, outputs, and analysis

## Citation

Dataset: Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. 
Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad 
E. H. Chowdhury (2020). "Reliable Tuberculosis Detection using Chest X-ray 
with Deep Learning, Segmentation and Visualization." IEEE Access, Vol. 8, 
pp. 191586-191601. DOI: 10.1109/ACCESS.2020.3031384.
