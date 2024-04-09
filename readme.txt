Deep-Kernel-Learning for STM

Deep Kernel Learning (DKL) method is used for automated discovery of structure-property correlation using a scanning tunneling microscope. The deep kernel consists of a deep neural network combined with a standard Gaussian Process (GP) kernel, such as a radial basis function (RBF). The DNN had three layers with the first, second, and third layers consisting of 64, 64, and 2 neurons respectively, with ReLu activation function in the first two layers. The output of the last layer was treated as inputs to the RBF kernel.
The DKL uses open-source python package AtomAI (https://github.com/pycroscopy/atomai) was used for image processing and feature extraction, while GPax (https://github.com/ziatdinovmax/gpax) was used for the DKL-based training and GP regression. These were integrated with LabView programs to access the STM controls.

Implementation

1. The LabVIEW program integrates the DKL implementation across the data analysis and the instrumentation control.

2. The python program "DKL_using_gpax.py" is used for the based regression and analysis. The function "dkl_gpax_LV" interfaces with the Labview program to parse control parameters.

3. The framework uses an additional "scalarizer.py" which is used to process spectoscopic information that is used to create the targets for model training.

4. For the autonomous DKL exploration, the program described in "next_DKL_coordinate.py" is used to determine the successive frame for DKL.


Experimental metadeta files

The h5 files contain the spectroscopy data for the DKL implemented on the STM.

The files are named as the figures in the manuscript. The frame size of the stm image is given in the file name

Each dataset contains the tunneling spectra. There are four channels: Bias, Current, didv (LIX), LIY.

 To access metadeta information for the eperimental descriptors for each of the iterations:

1. dataset.metadata['DKL_position']  # Provides the position descriptors for the iterations
2. dataset.metadata['DKL_scalar'] # Provides the scalar value for the iterations
3. dataset.metadata['scan_image'] # the 2D array containing the STM scan image
