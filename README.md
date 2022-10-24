# biosignalGANs
Adversarial learning models for biological signals including artificial synthesis and modality transfer. This repository provides code/implementation details of the models proposed in **Generalized Generative Deep Learning Models for Biosignal Synthesis and Modality Transfer.** 

![maindiagram](https://user-images.githubusercontent.com/19911856/197334132-fd593419-1e66-4bd7-b89b-4f23e613d6a1.png)

## Requirements:
1. Pytorch 1.5.0>
2. Numpy
3. Matplotlib 

## Databases: 
For our experiments, we use publically available datasets. We have given our feature extraction codes (*create_data.py*) in each corresponding repository. 
1. PTBXL (from PhysioNet) - https://physionet.org/content/ptb-xl/1.0.0/
2. EPHNOGRAM (from PhysioNet) https://physionet.org/content/ephnogram/1.0.0/
3. PhysioNet-CinC2016 - https://physionet.org/content/challenge-2016/1.0.0/
4. MIT-BIH Arrhythmia - https://www.physionet.org/content/mitdb/1.0.0/

## Steps for running our algorithms:
1. Run our segment generation codes after downloading the corresponding datasets. Please be careful about the directory structure in your own machine. Save the data properly and provide the exact location of the data while training the models. 
2. To load the data and prepare the data for a subject independent evaluation protocol, we often use dictionary structured files. These files will be automatically written into the corresponding folders if needed, and will be loaded when training. 
3. Most of our scripts have a visualization directory where we write the reconstructions/generations while training the models. We use this to manually check the performance of the GAN. The directory structure starts with .SIN, and you have to create that directory in your folder. 
4. Furthermore, DIR = T*/V* denotes the test number and the version number we used. When you’re training the model, your epoch-level visualizations will be written to ./SIN/T*/V*/f{epoch}/ and the model state will also be written into that folder. 
5. The loss variations will get saved in the Loss file. 
6. This repository provides all the implementation details for the models in the paper with exact pytorch modules and training functions. Our sole intention is to provide the code segments, and we encourage researchers to use their own customized scripts while following our data extraction protocols and loss functions. 

## Specific Details.
1. EPHNOGRAM
   - PixtoPix-style-transfer [PCG-to-ECG](./EPHNOGRAM/pcg_to_ecg.py)
   - PixtoPix-style-transfer [ECG-to-PCG](./EPHNOGRAM/ecg_to_pcg.py)
   - GAN for 3s ECG generation [ECG](./EPHNOGRAM/ecg_to_pcg.py)
2. MIT-BIH
   - Baseline model implementation DCGAN from [Paper](https://www.sciencedirect.com/science/article/pii/S0925231220306615) our implementation [code](./MIT-BIH/baseline.py)
3. PTB
   - PixtoPix-style-transfer [VCG-to-12lead ECG](./PTB/vcg_to_12ecg.py)
4. PTBXL
   - WAE-GAN for generation/reconstructions [12lead ECG]
   
   
