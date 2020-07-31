# Denoiser Comparator
Little helper framework to compare image denoisers. The structure is split in components that can be selected via command line arguments. The following component types are available:
* **denoiser**: Each denoising technic is implemented as an individual denoiser that can be selected at runtime
* **metric**: Noise reduction can be measured through various metrics (some priorize the color discrepancy, others take into account human eye perception, etc). A few different metrics can be enabled.
* **dataset**: Multiple image datasets can be implemented, but only one can be used at a time

## Structure

* `denoisers/` has the denoiser implementations
* `metrics/` has the metrics implemented
* `datasets/` has the datasets implemented

## Requirements

The best way to make sure you are running with all the required dependencies is to create a *venv* 
and install all the requirements there:
```bash
# create the venv
python3 -m venv name_of_venv

# activate the venv
source name_of_venv/bin/activate

# install the requirements
pip install -r requirements.txt
```

### Note on BM3D
Apart from the python modules installed via *pip*, if you want to use the **bm3d** denoiser, you need
to install the OpenBLAS development package (*openblas-devel* on OpenSUSE and Fedora, *libopenblas-dev*
on Debian and Ubuntu).

## Running
You can check the available command line arguments by calling:
```
./denoise_comparator.py --help
```

If you would like to see the available components, just invoke the main script with `--list`:
```
./denoise_comparator.py --list
```

Enjoy!
