# Fairness Exploration: Latent Feature Discover

This quarter we are focusing on the discovery of fair latent variables using an encoder-decoder model. This model will be used on on tabular datasets that have features, X, an identifiable sensitive feature, S, and observed labels, Y. We will see how this algorithm performs on common fairness datasets as well as 2 new datasets that we are further exploring with. 
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
We looked at synthetic data where we have true unbiased data and data with injected biased error
**To retrieve results for the Synethic data**
    -Run in terminal
   ```bash
   python run.py synthetic_binary
   ```

We then looked at real world data where the the sensitive and observed labels are binary values. For this portion, we looked 3 common datasets used to explore fairness applications: UCI Adults, German Credit Scores, and Compas. 

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



     