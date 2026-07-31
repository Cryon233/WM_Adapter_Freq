export PROJECT_ROOT="$HOME/control-frequency-wm"
export STORAGE_ROOT="$PROJECT_ROOT/storage/jepa-wms"

export JEPAWM_HOME="$PROJECT_ROOT/third_party"
export JEPAWM_DSET="$STORAGE_ROOT/datasets"
export JEPAWM_LOGS="$STORAGE_ROOT/logs"
export JEPAWM_OSSCKPT="$STORAGE_ROOT/opensource-checkpoints"

export JEPA_WM_DROID_CKPT="$STORAGE_ROOT/checkpoints/jepa_wm_droid.pth.tar"
export JEPAWM_ROBOCASA_HDF5="$JEPAWM_DSET/robocasa/combine_all_im256.hdf5"

export DINOV3_DIR="$JEPAWM_OSSCKPT/dinov3"
export DINOV3_VITL16_CKPT="$DINOV3_DIR/dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=0
