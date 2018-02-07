# RCAE_GMM: a radio galaxy morphology generator
This repo aims to construct a radio galaxy (RG) morphology generator by training a residual convolutional autoencoder (RCAE), and simulate new RG samples by feeding randomly generated features into the decoder subnet. The Gaussian mixture models are estimated for generating the new feature vectors. 

## Construction of the rcae\_gmm package
In summary, we provide classes for the RCAE network construction as well as some utilities for image preprocessing, network saving and restoration, and etc. Detailed instruction and usage please refer to the code files. Here we list the single python based scripts,
- [Bottleneck_en.py](https://github.com/myinxd/rcae_gmm/blob/master/rcae/bottleneck/bottleneck_en.py) and [Bottleneck_de.py](https://github.com/myinxd/rcae_gmm/blob/master/rcae/bottleneck/bottleneck_de.py): Classes for bottlenecks in the encoder and decoder;
- [Block.py](https://github.com/myinxd/rcae_gmm/blob/master/rcae/block/block.py): A class for blocks in the residual convolutional network, which is a box hosting multiple bottlenecks;
- [utils.py](https://github.com/myinxd/rcae_gmm/blob/master/rcae/utils/utils.py): Auxiliary utilities.

Notebooks are provide as demos for the user to construct their own residual convolutional networks, which are 
- [notebook-rcae-25.ipynb](): Construct a 25-layer (12 + 12 + 1: encoder+decoder+code_layer);
- [notebook-rcae-25-restore.ipynb](): Restore the trained network
- [notebook-rg-generation.ipynb](): Generate new radio galaxies

## Requirements
Some python packages should be installed before applying the nets, which are listed as follows,
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [astropy](https://www.astropy.org/)
- [matplotlib](http://www.matplotlib.org/)
- [Tensorflow](http://www.tensorflow.org/)
- [scikit-learn](http://scikit-learn.org/)

Also, [CUDA](http://develop.nvidia.org/cuda) is required if you want to run the codes by GPU, a Chinese [guide](http://www.mazhixian.me/2017/12/13/Install-tensorflow-with-gpu-library-CUDA-on-Ubuntu-16-04-x64/) for CUDA installation on Ubuntu 16.04 is provided.

## Usage
Before constructing a RCAE net, the pakcage should be installed. Here is the installation,
```sh
$ cd rcag-gmm
$ pip3 install --user .
```
Detailed usage of our rcae-gmm package is demonstrated in [demo]((https://github.com/myinxd/rcae_gmm/blob/master/demo/) by jupyter notebooks.


## Contributor
- Zhixian MA <`zx at mazhixian.me`>

## License
Unless otherwise declared:

- Codes developed are distributed under the [MIT license](https://opensource.org/licenses/mit-license.php);
- Documentations and products generated are distributed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US);
- Third-party codes and products used are distributed under their own licenses.
