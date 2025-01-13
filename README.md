# Data Heterogeneity - Q1

This quarter's work focused on exploring clustering methods for latent variables. We began by examining common clustering methods, such as **K-Means Clustering** and **Gaussian Mixture Models**, applied to the UCI Adult dataset and Kaggle's COVID-19 dataset. Building on this, we implemented a more complex **neural network-based model** to cluster the data and identify latent (unknown) variables. This algorithm was applied to four datasets: three tabular datasets and one image dataset.

**Note:** These datasets are quite large, and running them efficiently will likely require a GPU.

## Getting Started

To achieve the same results as we did, follow these steps:

### Clone the Repository

Clone this repository and navigate to the root directory.
```bash
git clone https://github.com/linabat/data-heterogeneity-q1.git
```

### Install Dependencies

Install the required dependencies by running the following command while in the project repository:

```bash
conda  create -n data-heterogeneity-q1 python=3.11.9
```
Once the environment has been created, run 
```bash
conda activate data-heterogeneity-q1
```
```bash
pip install -r requirements.txt
```

Let's start by looking at the results for the **waterbirds dataset**. This dataset comes from [this repository](https://github.com/kohpangwei/group_DRO). Below are the steps to get this code working for this dataset and retrieve the same results as we did
### Steps:
1. **Download the Waterbirds Dataset**  
   - [Click here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz) to download the dataset.
   - Save the `.tar.gz` file in the cloned repository directory and note the file's path.
2. **Update Configuration**  
   - Navigate to the `config` folder in the repository.
   - Update the `tar_file_path` key in the `waterbids_download_data.json` file with the relative path to the `.tar.gz` file.
3. **To run all the steps for waterbirds**
    -Run in terminal
   ```bash
   python run.py run_all_waterbirds
   ```
**NOTE: If an error about tensorflow.keras module not found run**

  ```bash
  pip install --upgrade --force-reinstall tensorflow
  ```
**To run the steps seperately**
3. **Extract the Dataset**  
    - Run the following command to extract the dataset into a folder called `waterbirds_data` in the repository directory:
     ```bash
     python run.py download_wb_data
     ```
4. **Generate Features**  
   - Process the dataset to retrieve the features required to run the model by executing:
     ```bash
     python run.py waterbirds_features
     ```
   - This will create a folder called `waterbirds_features` containing the `features.npy` file.
5. **Run the Model**  
   - To run the clustering model on the Waterbirds dataset, execute:
     ```bash
     python run.py waterbirds_run_data
     ```
   - This command will generate the following outputs:
     - **Clustered Images**: Stored in the `waterbirds_image_results` folder.
     - **Predicted Clusters CSV**: Stored in the `wb_retrieved_data` folder.
     - **Performance Metrics**: Results will be saved in the `model_results` folder.

---

### Results

Below are the results for the Waterbirds dataset:  
Accuracy Proportions Per Cluster\
Cluster 0: 0.9771\
Cluster 1: 0.5721

---
## Census-Based Datasets

Moving on, let's take a look at the other 3 datasets which come from the same [source](https://github.com/socialfoundations/folktables). 

Moving on, let's take a look at the other three datasets, which come from the same [source](https://github.com/socialfoundations/folktables). These datasets are derived from the Census dataset and use three different data processors. Each processor generates distinct models and clusters. The data processors are: 

- **ACSIncome**
- **ACSEmployment**
- **ACSPublicCoverage**

The steps to process these datasets are the same for all three data processors. Below, we'll use **ACSIncome** as an example.

### Steps for Processing ACSIncome
1. **Run All Results**
   To run all the relevant parts to retrieve the census results, run in terminal
   ```bash
   python run.py run_census_all
   ```
2. **Run the Model Individual Parts of the Model**
   Will be using the income dataset for this example, but this these are steps that can be replicated for the other 2 datasets below (mentioned below)
   Run in terminal
   ```bash
   python run.py census_income_model
   ```
    - This will output csv with the clusters for each row in `census_retrieved_data`
    - In `model_results` folder, will be able to see the states the belong to each cluster.
    - The number of clusters can be changed in the config file (in this case, the file is `config/census_income.json`)
4. **Visualize Cosine Similarity**
   To run cosine similarity to visually see the similarities between the states, run
   ```bash
   python run.py census_income_cosine
   ```
    - The output plot will be saved in the `census_image_results` folder

#### Running ACSEmployment and ACSPublicCoverage
Here are is how to run the other two data processor datasets 
```bash
python run.py census_employment_model 
```
```bath
python run.py census_employment_cosine
```

``` bash
python run.py census_public_coverage_model 
```
```bash
python run.py census_public_coverage_cosine
```
---

### Results
Below are the results for the Waterbirds dataset:  
The clusters for employment are: 
Cluster 0: = [AK, AL, AR, AZ, CA, CO, CT, FL, GA, IA, ID, IL, IN, KS, KY, LA, MA, MD, ME, MI, MN, MO, MS, MT, NC, ND, NH, NJ, NM, NV, NY, OH, OK, OR, PA, RI, SC, SD, TN, TX, UT, VA, VT, WA, WI, WV, WY]
Cluster 1: = [DE]
Cluster 2: = [HI, PR]

The clusters for income are: 

Cluster 0: = [AK, AL, AR, AZ, CO, CT, DE, FL, GA, IA, ID, IL, IN, KS]
Cluster 1: = [NC, ND, NH, NJ, NM, NY, OH, OK, OR, PA, RI, SC, SD, TN, TX, UT, VA, VT, WA, WI, WV, WY]
Cluster 2: = [CA, HI, NV, PR]
Cluster 3: = [KY, LA, MA, MD, ME, MI, MN, MO, MS, MT]

The clusters for public coverage are: 
Cluster 0: = [ND, OH, OK, OR, PA, PR, RI, SC, SD, TN, TX, UT, VA, VT, WA, WI, WV, WY]
Cluster 2: = [AK, AL, AR, AZ, CA, CO, CT, DE, FL, GA, HI, IA, ID, IL, IN, KS]
Cluster 3: = [KY, LA, MA, MD, ME, MI, MN, MO, MS, MT, NC, NH, NJ, NM, NV, NY]

---

### At the start of the quarter, we look into common GMM and KMeans clustering methods on UCI Adults dataset and a Covid Dataset 
Our results are stored in the `gmm_kmean_results` folder. If you would like to replicate the results for the Adults Dataset, in terminal run 
```bash
python run.py gmm_adults
```
```bash
python run.py kmeans_adults

```

If you would to replicate the results for the COVID dataset, you will first need to download the csv from [kaggle](https://www.kaggle.com/datasets/meirnizri/covid19-dataset) and save csv in the repository. Go to `config` folder and go to `covid.json`. From here, paste the path to the csv for `covid_fp key`.  The in terminal run  
```bash
python run.py gmm_covid
```
```bash
python run.py plot_gmm_covid
```

Despite the common use of these clustering algorithms, with the results retrieved, it can be seen that the performance of such model is not good, indicating that we should look into other algorithms for clustering.




     