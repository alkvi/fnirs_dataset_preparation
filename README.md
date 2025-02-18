## Park-MOVE fNIRS data preparation

This repository contains code to prepare raw data for the Park-MOVE fNIRS study.

The resulting dataset can found in the data repository at: https://doi.org/10.48723/vscr-eq07

### fNIRS data preparation 

[SNIRF](https://github.com/fNIRS/snirf) and [BIDS](https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/11-near-infrared-spectroscopy.html) output is produced from original NIRx format data, produced in Aurora 1.4 with a NIRx NIRsport 2 8x8 system.

- First, _convert_to_snirf.m_ is run. This generates SNIRF files using the NIRS Brain AnalyzIR toolbox. This was run on MATLAB R2021a with a forked version of the toolbox:
https://github.com/alkvi/nirs-toolbox-fork/commit/dfc4e28d2c8ed5abf4685a6b7a8e596a7d07a1e4
- Then, the BIDS output structure was created by running _prepare_bids_structure.py_.
- Then, SNIRF files were converted into a BIDS dataset using [sourcedata2bids](https://github.com/rob-luke/fnirs-apps-sourcedata2bids) (v0.4.5).
- Then, SNIRF files were validated using [pysnirf2](https://github.com/BUNPC/pysnirf2) (v0.7.3) by running _validate_snirf_files.py_.
- A folder with an original NIRx probeInfo file is created in nirx_probe.

### IMU data preparation

IMU data is captured with APDM Mobility Lab using sensors on feet and lumbar. Gait data is then extracted and analyzed per block.

- First, _prepare_apdm_data.py_ is run to extract raw data from HDF5 files into [Apache Parquet](https://parquet.apache.org/docs/) format files.
- Then, _prepare_block_points.py_ is run to extract stimuli block timepoints from the BIDS dataset.
- Then, _sync_imu_data.py_ is run to adjust synchronization timepoints between fNIRS and IMU data.

Spatiotemporal gait variables are then calculated by running _gait_variables_gaitmap.py_.