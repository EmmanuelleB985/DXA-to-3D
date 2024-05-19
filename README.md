# DXA_to_3D

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
We achieve this by predicting the coronal view (centerline and lateral curves) as well as sagittal view (centerline and lateral curves). Given 2 orthogonal projections of the 3D spine, we can reconstruct the 3D spine. This gives a user-friendly viusalisation of the spine that could be used in complement to 2D DXA to measure scoliosis. 


2 main contributions : 
* Regression of 3D patient-specific spine shapes from 2D AP DXA only. 
* Use of lightweight transformer and ResNet50 backbone surpassing Transformer and CNN-based models


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


## ToDo

- [x] Training code 
- [x] Documentation 
- [ ] Checkpoints 


## Acknowledgments

If you found this work useful, please cite the following papers: 


@InProceedings{Bourigault23,
  author       = "Emmanuelle Bourigault  and Amir Jamaludin and Emma Clark and Jeremy Fairbank and Timor Kadir and Andrew Zisserman",
  title        = "3D Shape Analysis of Scoliosis",
  booktitle    = "MICCAI Workshop on Shape in Medical Imaging ",
  year         = "2023",
  publisher    = "Springer",
  keywords     = "MRI · Spine Geometry · 3D/2D Correspondences",
}

@inproceedings{Windsor21,
   author    = {Rhydian Windsor and Amir Jamaludin and Timor Kadir and Andrew Zisserman},
   booktitle = {Proc. Medical Image Computing and Computer Aided Intervention (MICCAI)},
   title     = {Self-Supervised Multi-Modal Alignment for Whole Body Medical Imaging},
   date      = {2021},
 }