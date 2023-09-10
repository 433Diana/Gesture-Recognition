# Gesture Recognition
 An implementation of gesture recognition algorithm based on RISC-V SoC

**\[Sept 10, 2023\] Updates: Add comparison of accuracy between original CNN model and quantilization model.**

# Project structure
Matlab: Matlab project of gesture recognition inference model. Including model quantization and accuracy analysis.
 figure: Gesture recognition dataset used in this project.
 variable_*: Weights of model with different length of words.
 CNN_*.m: Matlab function to implement several layers in CNN model.
 compare.m: Comparison of accuracy before and after model quantization.
 extract.m: Extract parameters from hdf5 file.
 infer.m: Guess gesture from 1 input image.
 quan_model.hdf5: CNN model.

Python: Python project of training CNN model.
 imgfolder: Gesture recognition dataset used in this project. 
 model: Weights of CNN model after training (saved as hdf5 file).
 gesture.py: Python code of training CNN model.
 gesture3.yaml: Backup of Anaconda environment.

# Get Started
**Reminder:** Remember to edit parameters before running a project!!!

Create Anaconda environment using **gesture3.yaml**

Open Anaconda Prompt,
```Shell
conda env create -f gesture3.yaml
activate gesture3
```

Running model training,
```Shell
python gesture.py
```

Copy model file (*.hdf5) to ./Matlab

In Matlan command line window:

Extract weights parameter from hdf5 model,
```Shell
extract
```

Run one image inference,
```Shell
infer
```
or run accuracy analysis,
```Shell
compare
```

# Thanks
Abhishek Singh,”asingh33/CNNGestureRecognizer: CNNGestureRecognizer (Version 1.3.0)”, Zenodo. http://doi.org/10.5281/zenodo.1064825, Nov. 2017. 
