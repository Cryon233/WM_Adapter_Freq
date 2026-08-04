export LIBERO_ROOT="/absolute/path/to/LIBERO"
export LIBERO_DATA_ROOT="/absolute/path/to/libero/datasets"
export LIBERO_CONFIG_PATH="/absolute/path/to/libero/config"
# LIBERO is pinned to robosuite 1.4.0, independently of the newer RoboCasa
# fork. This directory must contain robosuite/__init__.py and its dependencies.
export LIBERO_ROBOSUITE_ROOT="/absolute/path/to/isolated/robosuite_1_4"
# Optional suite-specific overrides:
export LIBERO_SPATIAL_DATA_ROOT="/absolute/path/to/libero_spatial"
export LIBERO_GOAL_DATA_ROOT="/absolute/path/to/libero_goal"

export ROBOCASA_PLACE_HDF5="/absolute/path/to/jepa_wm_datasets/robocasa/combine_all_im256.hdf5"
# Preferred articulated source: complete task-level RoboCasa365 LeRobot v2.1
# data, including per-episode Parquet, videos, model.xml.gz, and states.npz.
export ROBOCASA_OPEN_DRAWER_LEROBOT="/absolute/path/to/robocasa365/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot"
# Optional official task-level HDF5 fallback:
export ROBOCASA_OPEN_DRAWER_HDF5=""

# cross_backend_adapter_v1 uses the official DINO-WM DROID predictor and the
# DINOv2 ViT-S/14 visual checkpoint. Both files are supplied locally; the suite
# never downloads model weights.
export DINO_WM_DROID_CKPT="/absolute/path/to/droid_dino-wm_noprop.pth.tar"
export DINOV2_VITS14_CKPT="/absolute/path/to/dinov2_vits14_pretrain.pth"

# Camera dimensions and the action/controller transform are discovered from the
# selected HDF5 and live official LIBERO environment; do not configure them here.
