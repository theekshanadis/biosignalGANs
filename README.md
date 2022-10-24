# biosignalGANs
Adversarial learning models for biological signals including artificial synthesis and modality transfer. This repository provides code/implementation details of the models proposed in **Generalized Generative Deep Learning Models for Biosignal Synthesis and Modality Transfer.** 

![maindiagram](https://user-images.githubusercontent.com/19911856/197334132-fd593419-1e66-4bd7-b89b-4f23e613d6a1.png)

Requirements:
1. Pytorch 1.5.0>
2. Numpy
3. Matplotlib 

Databases: For our experiments, we use publically available datasets. We have given our feature extraction codes (*create_data.py*) in each corresponding repository. 
1. PTBXL (from PhysioNet) - https://physionet.org/content/ptb-xl/1.0.0/
2. EPHNOGRAM (from PhysioNet) https://physionet.org/content/ephnogram/1.0.0/
3. PhysioNet-CinC2016 - https://physionet.org/content/challenge-2016/1.0.0/
4. MIT-BIH Arrhythmia - https://www.physionet.org/content/mitdb/1.0.0/
