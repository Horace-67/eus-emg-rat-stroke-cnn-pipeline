# EUS-EMG Sham vs MCAO Rat Classification Pipeline (MATLAB)

This repository contains a public MATLAB pipeline for EUS-EMG analysis in a rat study, including:

1. burst label loading from Excel
2. EUS-EMG preprocessing and activity gating
3. fixed-length window segmentation
4. STFT image generation
5. 1D CNN / 2D CNN leave-one-subject-out evaluation
6. rat-level probability pooling
7. stacking logistic regression fusion
8. example figure export

## Environment

- MATLAB R2024b
- Deep Learning Toolbox
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox

## Repository structure

```text
.
├─ eus_run_main_public.m
├─ README.md
├─ .gitignore
├─ labels/
│  └─ label_example.xlsx
└─ results/
   └─ example_outputs/
