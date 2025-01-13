import sys
import json
import os

# Add the 'src' directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

from etl import (
    retrieve_adult_data,
    gmm_adults,
    retrieve_covid_data,
    gmm_covid,
    plot_pca_gmm_covid,
    kmeans_adults,
    retrieve_wb_features,
    run_waterbirds, 
    run_census, 
    download_wb_data,
    run_census_cosine
    
)
    
if __name__ == '__main__':
    args = sys.argv[1:]
    # CENSUS
    if 'census_income_model' in args: 
        with open("config/census_income.json", "r") as file:
            config = json.load(file)
        run_census(**config)
        
    if 'census_income_cosine' in args: 
        with open("config/census_income_cosine.json", "r") as file:
            config = json.load(file)
        run_census_cosine(**config)
        
    if 'census_employment_model' in args: 
        with open("config/census_employment.json", "r") as file:
            config = json.load(file)
        run_census(**config)
        
    if 'census_employment_cosine' in args: 
        with open("config/census_employment_cosine.json", "r") as file:
            config = json.load(file)
        run_census_cosine(**config)

    if 'census_public_coverage_model' in args: 
        with open("config/census_public_coverage.json", "r") as file:
            config = json.load(file)
        run_census(**config)

    if 'census_public_coverage_cosine' in args: 
        with open("config/census_public_coverage_cosine.json", "r") as file:
            config = json.load(file)
        run_census_cosine(**config)

    if 'run_census_all' in args: 
        with open("config/census_income.json", "r") as file:
            config = json.load(file)
        run_census(**config)
        with open("config/census_income_cosine.json", "r") as file:
            config = json.load(file)
        run_census_cosine(**config)
        with open("config/census_employment.json", "r") as file:
            config = json.load(file)
        run_census(**config)
        with open("config/census_employment_cosine.json", "r") as file:
            config = json.load(file)
        run_census_cosine(**config)
        with open("config/census_public_coverage.json", "r") as file:
            config = json.load(file)
        run_census(**config)
        with open("config/census_public_coverage_cosine.json", "r") as file:
            config = json.load(file)
        run_census_cosine(**config)
        

    # WATERBIRDS
    if "download_wb_data" in args: 
        with open("config/waterbirds_download_data.json", "r") as file:
            config = json.load(file)
        download_wb_data(**config)
        
    if 'waterbirds_features' in args:
        retrieve_wb_features()

    if 'waterbirds_model' in args:
        with open("config/waterbirds_run_data.json", "r") as file:
            config = json.load(file)
        run_waterbirds(**config)

    if 'run_all_waterbirds' in args: 
        with open("config/waterbirds_download_data.json", "r") as file:
            config = json.load(file)
        download_wb_data(**config)

        retrieve_wb_features()

        with open("config/waterbirds_run_data.json", "r") as file:
            config = json.load(file)
        run_waterbirds(**config)

    # KMeans and GMMs
    if 'gmm_adults' in args: 
        with open('config/covid.json') as fh:
            data_params = json.load(fh)
        
        gmm_adults(data_params["gmm_adult_ts"])

    if 'gmm_covid' in args:
        with open('config/covid.json') as fh:
            data_params = json.load(fh)
        
        gmm_covid(
        data_params["covid_fp"],
        data_params["replace_num"],
        data_params["gmm_covid_ts"]
    )
        
    if 'plot_gmm_covid' in args: 
        with open('config/covid.json') as fh:
            data_params = json.load(fh)
            plot_pca_gmm_covid(
        data_params["covid_fp"],
        data_params["replace_num"],
        data_params["gmm_covid_ts"]
            )
    if 'gmm_test' in args: 
        with open('config/test_data.json') as fh:
            data_params = json.load(fh)
            gmm_covid(
        data_params["covid_fp"],
        data_params["replace_num"],
        data_params["gmm_covid_ts"], 
                test_data=True
    )

            plot_pca_gmm_covid(
        data_params["covid_fp"],
        data_params["replace_num"],
        data_params["gmm_covid_ts"], 
                test_data=True
            )
        print ("Test data completed")

    if 'kmeans_adults' in args:
        with open('config/covid.json') as fh:
            data_params = json.load(fh)
        kmeans_adults()
