# mwb_common
Python routines that I reuse, but mostly for the MFM project.

Here is a quick description of each module/file:

| Module | Description | 
|--------|-------------|
| stat_mwb.py| Calculates Cramer's V and Theil's U correlations; also includes a generalized undersampling routine that can handle "cohort" undersampling.| 
| mwb_bootstrap.py| Provides routines run a classifier multiple times with different initial random seeds with replacement. Optionally uses undersampling with the same seed. |
| util_mwb.py| Contains routines to simplify reading CSV datafiles                                                                                                       

