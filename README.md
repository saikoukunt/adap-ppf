# Adaptive Point Process Filter

### Installation
To install the package using pip: run `pip install -e .` in the same directory as `setup.py`. The package can then be imported as `adap_ppf`.

### Package Description
This package contains four functions related to adaptive point process filters:

- `ppf()`: Applies a basic point process filter to the input (spiking and behavioral) data
- `ofc_ppf()`: Applies a point process filter to the input (spiking) data using an optimal feedback control (OFC) framework. Intended for use with kinematic behavioral data
- `fit_vel_model()`: Uses maximum likelihood estimation to find parameters for a kinematic evolution model
- `fit_ofc_model()`: Computes paramters for an optimal feedback control model

More detailed documentation for each function is in `adap_ppf/filt.py`

Note: The data used in `scripts/Saccade Data.ipynb` is from `https://crcns.org/data-sets/pfc/pfc-7`
