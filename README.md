# Introduction

<p align="center">
  <img width="320" height="200" src="/assets/europvi1.jpg" hspace="30">
  <img width="320" height="200" src="/assets/europvi2.jpg" hspace="30">
</p>



This repository contains code for the paper,

[**Euro-PVI: Pedestrian Vehicle Interactions in Dense Urban Centers (CVPR 2021)**](https://openaccess.thecvf.com/content/CVPR2021/html/Bhattacharyya_Euro-PVI_Pedestrian_Vehicle_Interactions_in_Dense_Urban_Centers_CVPR_2021_paper.html)

[Apratim Bhattacharyya](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/apratim-bhattacharyya/), [Daniel Olmeda Reino](https://www.linkedin.com/in/danielolmeda/), [Mario Fritz](https://scalable.mpi-inf.mpg.de/), [Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele/)

# Requirements

The code is written in Python 3.6 and developed on CUDA 9.0.

Requirements:

- torch 1.6.0
- numpy 2.10.0
- tqdm 4.46.0

To install the requirements, follow the instructions below,

    conda config --add channels pytorch
    conda create -n <environment_name> --file requirements.txt
    conda activate <environment_name>


# Dataset

*Euro-PVI* needs to be downlaoded along with the trajectory data. Detailed installation instructions can be found here:

https://europvi.mpi-inf.mpg.de 

Note, that the user must agree to the terms and conditions for use of Euro-PVI.

# Training

Once the data has been downloaded, the Joint-\beta-cVAE model can be trained with,

    python main.py --data_root <path to Euro-PVI> 
    
Additionally, the batch size and number of training epochs can be set using the *batch_size* and *epochs* command line arguments. The *checkpoint_path* argument specifies the location of the saved model checkpoint.

# Evaluation

Pre-trained models are saved automatically in the ./ckpts/ folder (We will release pre-trainined models soon). To evalute checkpoints,
        
        python main.py --data_root <path to Euro-PVI> --from_checkpoint --checkpoint_path <path to checkpoint> 
    