# Learned Spectral Unmixing

This code supplements the paper "TODO" by TODO et al.

This code will be integrated within the PATATO toolbox and maintained there.


## Installation

We recommend using Anaconda as the base Python interpreter
for installing the code in Windows.

Run the following command to install the requirements in the versions
that we used for this work:

    pip install -r requirements.txt

Manually install jaxlib from https://github.com/cloudhan/jax-windows-builder

For my install on windows 10, I specifically used

    pip install "jax[cpu]===0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

Manually install cudnn libraries if you want to run tensorflow on the GPU.

On linux, the installation might be a little more straightforward.