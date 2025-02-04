# Fairness Exploration: Latent Feature Discover

This quarter's work focused on exploring clustering methods for latent variables. We began by examining common clustering methods, such as **K-Means Clustering** and **Gaussian Mixture Models**, applied to the UCI Adult dataset and Kaggle's COVID-19 dataset. Building on this, we implemented a more complex **neural network-based model** to cluster the data and identify latent (unknown) variables. This algorithm was applied to four datasets: three tabular datasets and one image dataset.

**Note:** These datasets are quite large, and running them efficiently will likely require a GPU.

## Getting Started

To achieve the same results as we did, follow these steps:

### Clone the Repository

Clone this repository and navigate to the root directory.
```bash
git clone https://github.com/linabat/fairness-application.git
```

### Install Dependencies

Install the required dependencies by running the following command while in the project repository:

```bash
conda  create -n fairness-application python=3.11.9
```
Once the environment has been created, run 
```bash
conda activate fairness-applications
```
```bash
pip install -r requirements.txt
```
We first started looking at data where the the sensitive and observed labels are binary values. For this portion, we looked 3 common datasets used to explore fairness applications: UCI Adults, German Credit Scores, and Compas. 

Let's start by looking at the results for the **UCI adults dataset**. You can find this dataset[here](https://archive.ics.uci.edu/dataset/2/adult). 
**To retrieve results for UCI Adults**
    -Run in terminal
   ```bash
   python run.py adult_data
   ```

**To retrieve results for German Credit Score**
    -Run in terminal
   ```bash
   python run.py german_data
   ```

**To retrieve results for Compas**
    -Run in terminal
   ```bash
   python run.py compas_data
   ```

*The results for all 3 of these datasets will be outputed in the **model_results** folder*
The **model_results** folder will contain two different types of file. 
    -**.txt** which will contain the log messages for the different datasets that were run 
    -**.png** which will contain a visualization that shows the comparison between the baseline and adversarial model


**NOTE: If an error about tensorflow.keras module not found run**

  ```bash
  pip install --upgrade --force-reinstall tensorflow
  ```
---



     