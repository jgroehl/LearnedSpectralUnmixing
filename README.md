# Code supporting the paper: "Distribution-informed and wavelength-flexible data-driven photoacoustic oximetry"

This code supplements the paper **"Distribution-informed and wavelength-flexible data-driven photoacoustic oximetry"** 
by *Janek Gröhl, Kylie Yeung, Kevin Gu, Thomas R. Else, Monika Golinska,
Ellie V. Bunce, Lina Hacker, and Sarah E. Bohndiek*.

While this repository will remain a snapshot at the time point of paper submission (except for major bug fixes), 
the functional parts code will 
be integrated within the [PATATO toolbox](https://github.com/BohndiekLab/patato) and maintained there.


## Installation

We recommend using Anaconda as the base Python interpreter
for installing the code in Windows.

Run the following command to install the requirements in the versions
that we used for this work:

    pip install -r requirements.txt

Manually install jaxlib from https://github.com/cloudhan/jax-windows-builder

For my install on windows 10, we specifically used

    pip install "jax[cpu]===0.3.14" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

Manually install cudnn libraries if you want to run tensorflow on the GPU.

On linux, the installation might be a little more straightforward but we did not specifically test this.

