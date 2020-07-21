# Denoiser Comparator
Little helper framework to compare image denoisers. The structure is split in components that can be selected via command line arguments. The following component types are available:
* **denoiser**: Each denoising technic is implemented as an individual denoiser that can be selected at runtime
* **metric**: Noise reduction can be measured through various metrics (some priorize the color discrepancy, others take into account human eye perception, etc). A few different metrics can be enabled.
* **dataset**: Multiple image datasets can be implemented, but only one can be used at a time

## Structure

* `denoisers/` has the denoiser implementations
* `metrics/` has the metrics implemented
* `datasets/` has the datasets implemented

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
