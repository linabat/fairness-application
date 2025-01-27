import pandas as pd 
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import fetch_openml
import seaborn as sns

from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, 
    MaxPooling2D, BatchNormalization, Dropout
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler
)
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import Adam
from keras.initializers import Constant

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
import pandas as pd
from tqdm import tqdm
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, 
    MaxPooling2D, BatchNormalization, Dropout
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler
)

from tensorflow.keras.regularizers import Regularizer

from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import os
import tarfile

# This will be used when saving the files
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_pcat_compass(raw_data_path): 
    df = pd.read_csv('raw_data_path', 
                     parse_dates = ['DateOfBirth'])
    
    removed_columns = [
        'Person_ID',
        'AssessmentID',
        'Case_ID',
        'LastName',
        'FirstName',
        'MiddleName',
        'Screening_Date',
        'RecSupervisionLevelText',
        'RawScore',
        'DecileScore',
        'IsCompleted',
        'IsDeleted'
    ]
    df.drop(removed_columns, axis=1, inplace=True)

    age = ((pd.Timestamp.now() - df['DateOfBirth']).dt.days / 365.25).astype(int)
    age[age<0] = np.nan
    df['age_'] = age
    
    
    df.drop(df[df['ScoreText'].isnull()].index, inplace=True)
    df.drop(df[df['age_'].isnull()].index, inplace=True)
    df.drop(df[df['MaritalStatus']=='Unknown'].index, inplace=True)
    
    
    age_bins = [0, 30, 100]
    age_groups = pd.cut(df['age_'], bins=age_bins)
    df['Age'] = age_groups
    num_groups = len(df['Age'].cat.categories)
    df['Age'] = df['Age'].cat.rename_categories(range(num_groups))
    
    
    df.Sex_Code_Text = pd.Categorical(df.Sex_Code_Text)
    df['Sex_Code_Text'] = df.Sex_Code_Text.cat.codes

    df["Ethnic_Code_Raw"] = df["Ethnic_Code_Text"]
    df.Ethnic_Code_Text = pd.Categorical(df.Ethnic_Code_Text)
    df['Ethnic_Code_Text'] = df.Ethnic_Code_Text.cat.codes
    
    df.MaritalStatus = pd.Categorical(df.MaritalStatus)
    df['MaritalStatus'] = df.MaritalStatus.cat.codes
    
    df.CustodyStatus = pd.Categorical(df.CustodyStatus)
    df['CustodyStatus'] = df.CustodyStatus.cat.codes
    
    df.LegalStatus = pd.Categorical(df.LegalStatus)
    df['LegalStatus'] = df.LegalStatus.cat.codes
    
    df['ScoreText_'] = -1
    mask = df['ScoreText']=='High'
    df.loc[mask, 'ScoreText_'] = 0
    df.loc[~mask, 'ScoreText_'] = 1
    
    
    df.drop(['DisplayText','ScoreText','Agency_Text', 'AssessmentType', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason', 'Language'], axis=1, inplace=True)
    df.drop(['DateOfBirth', 'age_', 'Scale_ID'], axis=1, inplace=True)
    
    
    df.to_csv('data/compass_categorized.csv')


# Compass Pre-Processing

def load_compas_data_binarized(data_path):
    """
    Retrieving X, s and d from original cleaned data 
    Ethnicity is 0 for Caucasian, 1 for non-caucasian
    """
    df = pd.read_csv(data_path)    

    # "Ethinic_Code_Text" is being used as the sensitive feature
    # Observed label D is "ScoreText_"

    labels = df["ScoreText_"]
    features = df[~df["ScoreText_"]

    # adding a column that indicates what type
    def get_ethn(val):
        if val == 0: 
            return "Caucasian"
        else: 
            return "Other"

    df["ethn_from_df"] = df.apply(get_ethn, axis = 1)

    return features, labels, ethn_from_df

### CHANGE - BUT COME BACK TO AT THE END 
def load_compas_data_pcategorizing(data_path):
    df = pd.read_csv(data_path)    
    ethn_from_df = df["Ethnic_Code_Raw"] 
    labels = df["ScoreText_"]
    features = df[~df["ScoreText_", "Ethnic_Code_Raw"]]

    # adding a column that indicates what type
    
    return features, labels, ethn_from_df
 
# Clustering method - Parjanya's code 

# ===========================
# Helper Classes
# ===========================

class ClipConstraint(Constraint):
    """
    Clips model weights to a given range [min_value, max_value].
    Normalizes weights along a specific axis to exurethey sum to1
    """
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, weights):
        w = tf.clip_by_value(weights, self.min_value, self.max_value)
        return w / tf.reduce_sum(w, axis=1, keepdims=True)
    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class VarianceRegularizer(Regularizer):
    """
    Custom regularizer for maximum weight variance.
    Purpose: encourage uniformity among weights -- improve generalization or stability
    """
    
    def __init__(self, factor=0.01):
        self.factor = factor
    
    def __call__(self, x):
        variances = tf.math.reduce_variance(x, axis=1)
        max_variance = tf.reduce_max(variances)
        return self.factor * max_variance
    
    def get_config(self):
        return {'factor': self.factor}

# ===========================
# Model Definition and Training
# ===========================

## add sensitvite variable - need independence between sensitive and latent are independent -- modify the training 

### DON'T CHANGE
def lr_schedule(epoch):
    """Defines the learning rate schedule."""
    if epoch < 20:
        return 1e-4
    else:
        return 1e-5



# need to directly enforce independence between the observed decision variable D and senstivie attributes S  -- can enfource indpendent by setting the latent variable D_f to 
# always be equal to D -- results in marginal distribution over S, X, D 
# D is the outcome that is influence by S and X


# def enforce_independent(X,s,d):  ### DO WE EVEN NEED THIS???
    
    
### CHANGE
# WE HAVE X -- ALL THE FEATURES 
# s = proxy variable 
# n_z = number of clusters
# need to add D = observed variable, D_f is the latent variable that we adjust for 


# IN THIS CASE d is our proxy 
# S is the new senstive variable that we are adding in and then we need to establish indpendence with s and d?????? cause there was an assumption that was mde that D_f = D, which enforces the indpendence of D and S 

# D is ScoreText_  which is binary 
# S is Ethinic_Code_Text which is also binary 

# Need to add independence regularizer 
# get the latent representation for target_layer and make it independent from S 
## adversarial learning -- need to add regularizer to the loss function

### our case ScoreText is S (proxy) 
# def get_model_z(X,d,s,n_z,model_name,epochs=20,verbose=1,var_reg=0.0): # wrong header

## pass the code in as is 
## READ THE PAPER FIRST 

def get_model_z(X,s,n_z,model_name,epochs=20,verbose=1,var_reg=0.0):
    """
    Defines and trains a clustering model. 
    """
    # Ensuring that d and s are independent of each other 
    
    #Shuffle X and s together
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1) # added this in here isntead
    
    X = X[indices]
    s = s[indices]
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(s.shape[1], activation='linear'),
        Dense(n_z, activation='softmax', name='target_layer'), ## THIS IS THE LAYER THAT WE NEED TO GRAB TO ESTABLISH THE INDEPENDENCE
        Dense(s.shape[1], activation='linear', use_bias=False, kernel_initializer=RandomUniform(minval=0, maxval=1),kernel_constraint=ClipConstraint(0, 1), kernel_regularizer=VarianceRegularizer(factor=var_reg)), 
    ])
    optimizer = Adam(learning_rate=1e-3) # an adaptive learning rate optimizer
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model_checkpoint_path = os.path.join(repo_root, model_name)

    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=verbose
    )
    
    model.fit(
        X,
        s,
        batch_size=1024,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[model_checkpoint_callback, lr_scheduler]    
    )
    
    best_model = load_model(model_checkpoint_path, custom_objects={'ClipConstraint': ClipConstraint,'VarianceRegularizer': VarianceRegularizer})
    return best_model

# CHANGE
def pzx(X,best_model,arg_max=True):
    """
    Predict cluster assignments
    """
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    if(arg_max):
        p = np.argmax(p,axis=1)
    return p

### DON'T CHANGE
def set_seed(seed_num): 
    # Set random seed for reproducibility
    seed = seed_num
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ===========================
# Census Data Set
# ===========================

# CHANGE 
def run_compas(data_path, output_csv_name, output_model_results, num_clusters=2, num_epochs=40, model_path='best_census_model_inc.h5', num_var_reg=0, seed_num=0):
    
    set_seed(seed_num)

    features, labels, ethn_from_df = load_compas_data_binarized(data_path)
    # labels are already binarized
    best_model = get_model_z(features, labels, num_clusters, model_path, num_epochs, num_var_reg)

    p1_tr = pzx(features, best_model, arg_max=True)

    
    cluster_eg_df = pd.DataFrame({
        "cluster":p1_tr,
        "ethnic_group":ethn_from_df, 
        })

    # Save the DataFrame to a CSV file
    output_folder = os.path.join(repo_root, "compass_binarized_retrieved_data")
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, output_csv_name)
    cluster_state_df.to_csv(output_csv_path, index=False)

    # Group by 'ethinic group' and 'cluster', then count the occurrences of each cluster for each ethnic grouup
    cluster_counts = cluster_eg_df.groupby(["ethnic_group", "cluster"]).size().reset_index(name='count')
    
    # Calculate the total count for each 
    eg_totals = eg_state_df.groupby("ethnic_group")["cluster"].count().reset_index(name='total')
    
    # Merge the counts with the total counts per state
    merged = pd.merge(cluster_counts, state_totals, on="ethnic_group")
    
    # Calculate the proportion for each cluster within each state
    merged['proportion'] = merged['count'] / merged['total']
    result = merged.loc[merged.groupby("ethnic_group")["proportion"].idxmax()]
    cluster_groups = result.groupby("cluster")["ethnic_group"].apply(list)

    output_model_folder = os.path.join(repo_root, "model_results")
    os.makedirs(output_model_folder, exist_ok=True)
    output_txt_path = os.path.join(output_model_folder, output_model_results)
    with open(output_txt_path, "w") as file:
        file.write("Ethnic Group - Cluster Mapping Based on Highest Proportion:\n")
        file.write("=" * 50 + "\n\n")
        for cluster, states in cluster_groups.items():
            eg_list = ", ".join(ethnic_group)
            file.write(f"Cluster {cluster}: = [{eg_list}]\n")
    
    