Deep-Kernel-Learning for STM

Deep Kernel Learning (DKL) is a Bayesian deep learning approach that combines the deep neural networks (DNNs) with Gaussian Processes (GPs) for high dimentional (eg: images) data modeling. This method has been applied to scanning tunneling microscopy (STM) to automate the discovery of correlations between structure and properties in experimental datasets.

In this study, a three-layer DNN with 64, 64, and 2 neurons in the respective layers and ReLU activation in the first two layers is used. The final layer’s output serves as input to a Radial Basis Function (RBF) kernel, which is part of the GP model. The DKL model is trained on STM spectroscopy data, using the AtomAI library (https://github.com/pycroscopy/atomai) for image processing and GPax (https://github.com/ziatdinovmax/gpax) for DKL training and GP regression. Additionally, LabVIEW is employed for instrument control, allowing direct communication between the DKL model and STM hardware.

A simulated version of the DKL workflow can be run using the "Workflow_DKL_STM.ipynb" notebook, which simulates the DKL experiment on ground-truth grid data.


Experimental Details

1. The STM instrument is operated through a LabVIEW program, "DKL_Labview_interface.vi," which integrates the DKL model's decision-making process with STM control. The labview program uses subvis provided by the Nanonis programming interface library in addition to custom built subvis for python-script execution.

2. The core of the DKL implementation is managed through the Python script "DKL_using_gpax.py," which uses the GPax library for DKL-based GP regression. The collected STM images and spectroscopy data are processed using the AtomAI Python library, which handles image pre-processing and feature extraction. The regression model is trained on the STM image and spectroscopy data to predict properties of interest.

3. The file "scalarizer.py" is used to process the spectroscopic data, generating target scalar values for training the DKL model. Pre-processed data and metadata (e.g., DKL positions and scalar values) are stored for easy retrieval via labview functions.

4. An autonomous exploration algorithm in "next_DKL_coordinate.py" determines the next region for the STM to scan based on the GP regression results, optimizing the discovery of structure-property correlations.




Experimental metadeta files

The h5 files contain the spectroscopy data for the DKL implemented on the STM. An example of file handling is shown in "Analyze_DKL_data.ipynb"

The files are named as the figures in the manuscript. The frame size of the stm image is given in the file name

Each dataset contains the tunneling spectra. There are four channels: Bias, Current, didv (LIX), LIY.

 To access metadeta information for the eperimental descriptors for each of the iterations:

1. dataset.metadata['DKL_position']  # Provides the position descriptors for the iterations
2. dataset.metadata['DKL_scalar'] # Provides the scalar value for the iterations
3. dataset.metadata['image_attr'] # to access stm image attributes
