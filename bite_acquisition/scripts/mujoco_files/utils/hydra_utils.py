import hydra
from omegaconf import DictConfig, OmegaConf
import os
import glob

def load_saved_config_with_overrides(run_dir):
    # Path to the .hydra config file
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    overrides_path = os.path.join(run_dir, ".hydra", "overrides.yaml")
    
    # Load the saved configuration
    saved_cfg = OmegaConf.load(config_path)
    
    # Load the overrides
    overrides = OmegaConf.load(overrides_path)
    
    # Re-apply overrides
    for override in overrides:
        OmegaConf.update(saved_cfg, override.split('=')[0], override.split('=')[1])

    return saved_cfg

def find_latest_run_dir(base_dir, contain_string=None):
    # Find all directories in the base_dir
    all_subdirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]

    # Filter directories that contain the contain_string
    if contain_string is not None:
        all_subdirs = [d for d in all_subdirs if contain_string in d]
    # Sort the directories by creation time
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return latest_subdir