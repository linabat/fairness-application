import sys
import json
import os

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from etl import (
    main_binary,
    main_synthetic
)
    
if __name__ == '__main__':
    args = sys.argv[1:]
    # CENSUS
    if 'adult_data' in args: 
        with open("config/adult.json", "r") as file:
            config = json.load(file)
        main_binary(**config)

    if 'german_data' in args: 
        with open("config/german.json", "r") as file:
            config = json.load(file)
        main_binary(**config)

    if 'compas_data' in args: 
        with open("config/compas.json", "r") as file:
            config = json.load(file)
        main_binary(**config)

    if 'synthetic_binary' in args: 
        with open("config/synthetic_binary.json", "r") as file:
            config = json.load(file)
        main_synthetic(**config) ## need to add the synthetic config