# Applications of Fairness: Label Bias and Recovery of Ground Truth

Our capstone project focused on the discovery of ground truth labels from lablel bias using an encoder-decoder model. This model will be used on tabular datasets that have features, X, an identifiable sensitive feature, S, and observed labels, Y. We will see how this algorithm performs on common fairness datasets as well as 2 new datasets that we are further exploring with. 
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
conda activate fairness-application
```
Ensure you are in this repo's directory. 
```bash
pip install -r requirements.txt
```

We evaluated our model on two different scenarions. The first scenario being where the observed label is binary. The second scenario is where the observed label is multi-class. In both scenarios, the sensitive feature is binary. 

#### Binary Dataset Results
We looked at synthetic data where we have true unbiased data and data with injected biased error
**To retrieve results for the Synethic data**
    -Run in terminal
   ```bash
   python run.py synthetic_binary
   ```

We then looked at real world data where the the sensitive and observed labels are binary values. For this portion, we looked 3 common datasets used to explore fairness applications: UCI Adults, German Credit Scores, and Compas. 

Let's start by looking at the results for the **UCI adults dataset**. You can find this dataset [here](https://archive.ics.uci.edu/dataset/2/adult). 
**To retrieve results for UCI Adults**
    -Run in terminal
   ```bash
   python run.py adult_data
   ```

**To retrieve results for German Credit Score** You can find this dataset [here](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
    -Run in terminal
   ```bash
   python run.py german_data
   ```

**To retrieve results for Compas** You can find this dataset [here](https://github.com/propublica/compas-analysis). 
    -Run in terminal
   ```bash
   python run.py compas_data
   ```

#### Multi-Class Dataset Results

We looked also looked synthetic data where we have true unbiased data and data with injected biased error.
**To retrieve results for the Synethic data**
    -Run in terminal
   ```bash
   python run.py synthetic_multiClass
   ```

We then looked at a real world dataset where the the sensitive and observed labels are multi-class values. This is to evalute how well or code is working.  
**To retrieve results for Drug Consumption - Cannabis** You can find this dataset [here](https://www.kaggle.com/datasets/obeykhadija/drug-consumptions-uci). 
    - Download this dataset as a csv, name it 
    ``` bash
    "drug_consumption.csv"  
    ```
    - Create and place the csv in a "data" folder in the main directory of this repo
    -Run in terminal
   ```bash
   python run.py compas_data
   ```
### Results
*The results for all these datasets will be outputed in the **model_results** folder*
The **model_results** folder will contain two different types of file. 
    -**.txt** which will contain the log messages for the different datasets that were run, including metric values
    -**.png** which will contain visualizations that shows the comparison between the baseline and adversarial model. There are 2 visualizations that are outputted per dataset. The first visualization will be direct comparison between fairness and a confidence evaluation. The second visualization will include the all 4 evalatuion metrics we used for general evaluation. 

**NOTE: If an error about tensorflow.keras module not found run**

  ```bash
  pip install --upgrade --force-reinstall tensorflow
  ```

---



     