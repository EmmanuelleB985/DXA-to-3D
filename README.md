# 3D Spine Shape Estimation from Single 2D DXA (MICCAI2024 Oral)

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">Project Description</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

In this project, we aim to understand the 3D geometry of the spine at a patient-specific level given a single 2D DXA scan. For instance, scoliosis involves deformations in multiple directions with known deformations on the coronal, sagittal and axial plane. A better understanding of the deformations will lead to tailored treatments and better management of patients.
We propose an automated general framework to estimate the 3D spine shape from 2D DXA scans.
We achieve this by predicting the coronal view (centerline and lateral curves) as well as sagittal view (centerline and lateral curves). Given 2 orthogonal projections of the 3D spine, we can reconstruct the 3D spine. This gives a user-friendly visualisation of the spine that could be used in complement to 2D DXA to measure scoliosis. 


2 main contributions : 
* Regression of 3D patient-specific spine shapes from 2D AP DXA only. 
* Use of lightweight transformer and ResNet50 backbone surpassing Transformer and CNN-based models

<video src="assets/SPINE2D3D.mp4"width="320" height="240"></video>

## Getting Started
 
Please follow the below instructions to run the code.


### Prerequisites

1. Download the paired DXA-MRI data used in this work 

From the UKBiobank, this dataset can be downloaded after creating an account and registering on the UKBiobank platform. 
Follow instructions from this repo: 
https://github.com/rwindsor1/UKBiobankDXAMRIPreprocessing

2. Install the required packages 

```
pip install -r requirements.txt
```

## Usage

To train the model for spine curves regression, run 

```
python train.py
```
Can now be trained on a single GPU 10GB VRAM! Training time ~6hrs


You can download pre-trained model checkpoints [here] (https://www.dropbox.com/scl/fi/be4dg1xccgl1fo9wn74i8/epoch-996-loss_valid-points-best_loss-0.0168.pt?rlkey=ytnrrctofyebqtkj5p4554px1&st=zujv8ure&dl=0) and place in the checkpoints folder.

At inference time to obtain orthogonal spine curves from DXA scan, run 

```
python test.py
```

## ToDo

- [x] Training code 
- [x] Testing code 
- [x] Documentation 
- [ ] Demo coming soon


## Acknowledgments

If you found this work useful, please cite the following papers: 

```
@InProceedings{10.1007/978-3-031-72086-4_1,
author="Bourigault, Emmanuelle
and Jamaludin, Amir
and Zisserman, Andrew",
editor="Linguraru, Marius George
and Dou, Qi
and Feragen, Aasa
and Giannarou, Stamatia
and Glocker, Ben
and Lekadir, Karim
and Schnabel, Julia A.",
title="3D Spine Shape Estimation from Single 2D DXA",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="3--13",
isbn="978-3-031-72086-4"
}

@inproceedings{Windsor21,
   author    = {Rhydian Windsor and Amir Jamaludin and Timor Kadir and Andrew Zisserman},
   booktitle = {Proc. Medical Image Computing and Computer Aided Intervention (MICCAI)},
   title     = {Self-Supervised Multi-Modal Alignment for Whole Body Medical Imaging},
   date      = {2021},
 }
```
