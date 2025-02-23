import sys
import json
import os

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from etl import (
    main_binary,
    main_synthetic,
    cross_validation_data
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
        main_synthetic(**config) ## the configs added are the ones after doing cross validation

    if 'cv_adults' in args: 
        with open("config/cv_adults.json", "r") as file:
            config = json.load(file)
        cross_validation_data(**config) ## the configs added are the ones after doing cross validation

    if 'cv_german' in args: 
        with open("config/cv_german.json", "r") as file:
            config = json.load(file)
        cross_validation_data(**config) ## the configs added are the ones after doing cross validation

    if 'cv_compas' in args: 
        with open("config/cv_compas.json", "r") as file:
            config = json.load(file)
        cross_validation_data(**config) ## the configs added are the ones after doing cross validation



