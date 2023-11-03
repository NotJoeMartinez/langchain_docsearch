
import os

def get_or_create_config_path():
    # Construct the path to the directory
    config_path = os.path.expanduser('~/.config/docs')
    
    # Check if the directory exists
    if not os.path.exists(config_path):
        # If the directory does not exist, create it
        os.makedirs(config_path)
    
    # Return the path to the directory
    return config_path