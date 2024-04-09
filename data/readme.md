The h5 files contain the spectroscopy data for the DKL implemented on the STM.

The files are named as the figures in the manuscript. The frame size of the stm image is given in the file name

Each dataset contains the tunneling spectra. There are four channels: Bias, Current, didv (LIX), LIY.

 To access metadeta information for the eperimental descriptors for each of the iterations:

1. dataset.metadata['DKL_position']  # Provides the position descriptors for the iterations
2. dataset.metadata['DKL_scalar'] # Provides the scalar value for the iterations
3. dataset.metadata['scan_image'] # the 2D array containing the STM scan image
