import sys
import json
import os

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from etl_q2 import (
    load_compas_data_binarized

)
    
if __name__ == '__main__':
    args = sys.argv[1:]
    # CENSUS
    if 'census_income_model' in args: 
        with open("config/census_income.json", "r") as file:
            config = json.load(file)
        run_census(**config)