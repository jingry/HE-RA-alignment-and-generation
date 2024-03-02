# HE-RA-alignment-and-generation
# About this repository
This repository is for peer review of work "Generative Artificial Intelligence for In Silico Histopathology Image Synthesis from Raman Imaging as an Alternative to Intraoperative Assessment of Tongue Squamous Cell Carcinoma".
# Before using
Pleasee unzip the archived files, including 'data.zip', 'bestSataDict_UNET 128p.zip' and 'bestSataDict_ImgAlignNet.zip'. After unzip processes, please ensure the files and folders exists:
```
-Root
|-bestSataDict_UNET 128p.pt
|-bestSataDict_ImgAlignNet.pt
|-Notebook for classification task from ImgAlignNet.ipynb
|-Notebook for diffusion model.ipynb
|-repo_AE.py
|-repo_SD_DDPM.py
|-Data
  |-HE
    |-Normal
      |-N-TR-001 200.jpg
    |-Cancer
      |-CA-TR-001 200.jpg
  |-RAOriFP16
    |-Normal
      |-1-1-1.pt
      |-1-1-2.pt
      |-1-1-3.pt
      |-1-1-4.pt
      |-1-2-1.pt
      |-1-2-2.pt
      |-1-2-3.pt
      |-1-2-4.pt
    |-Cancer
      |-1-1-1.pt
      |-1-1-2.pt
      |-1-1-3.pt
      |-1-1-4.pt
      |-1-2-1.pt
      |-1-2-2.pt
      |-1-2-3.pt
      |-1-2-4.pt
```
# Environment
All the packages below can be installed using anaconda.
```
pytorch 2.0+ (torch and torchvision)
PIL
abc (Abstract Base Classes)
matplotlib
jupyter
```
# Usage
The two jupyter notebook files 'Notebook for classification task from ImgAlignNet.ipynb' and 'Notebook for diffusion model.ipynb' contains the related information of how ImgAlignNet works and how the H&E images generated from noise.
# Others
Only the data of the first patient is provided, please connect the author for all data.
