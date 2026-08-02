export LIBERO_ROOT="/absolute/path/to/LIBERO"
export LIBERO_DATA_ROOT="/absolute/path/to/libero/datasets"
export LIBERO_CONFIG_PATH="/absolute/path/to/libero/config"
# Optional suite-specific overrides:
export LIBERO_SPATIAL_DATA_ROOT="/absolute/path/to/libero_spatial"
export LIBERO_GOAL_DATA_ROOT="/absolute/path/to/libero_goal"

export ROBOCASA_PLACE_HDF5="/absolute/path/to/jepa_wm_datasets/robocasa/combine_all_im256.hdf5"
# Preferred articulated source. It must include task Parquet/video metadata and
# full simulator reconstruction metadata; policy-only LeRobot exports fail strict preflight.
export ROBOCASA_OPEN_DRAWER_LEROBOT="/absolute/path/to/robocasa365/OpenDrawer"
# Optional official task-level HDF5 fallback:
export ROBOCASA_OPEN_DRAWER_HDF5=""

# Camera dimensions and the action/controller transform are discovered from the
# selected HDF5 and live official LIBERO environment; do not configure them here.
