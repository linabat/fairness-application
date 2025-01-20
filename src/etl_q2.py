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

from tensorflow.keras.datasets import mnist
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
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.optimizers import Adam
from keras.initializers import Constant

from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

from tensorflow.keras.regularizers import Regularizer

from tensorflow.keras.regularizers import Regularizer 
from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.initializers import RandomUniform
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


# Compass Pre-Processing

def load_compas_data_binarized(data_path):
    
    df = pd.read_csv(data_path, 
                     parse_dates = ['DateOfBirth'])    
    # Drop these case-specific columns:
    # - Person_ID
    # - AssessmentID
    # - Case_ID
    # - LastName
    # - FirstName
    # - MiddleName
    # 
    # Drop these columns:
    # - `Screening_Date`: We don't know how to categorize this
    # - `RecSupervisionLevelText`: same as `RecSupervisionLevel`
    # - `RawScore`: many values. seems to be used for computation of target value
    # - `DecileScore`: seems to be used for computation of target value
    # - `IsCompleted`: same value for everyone
    # - `IsDeleted`: same value for everyone
        
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

    age = (datetime.datetime.now() - df.DateOfBirth).astype('timedelta64[Y]')
    age = age.astype('int')
    age[age<0] = np.nan
    df['age_'] = age
    
    # dealing with missing values
    df.drop(df[df['ScoreText'].isnull()].index, inplace=True)
    df.drop(df[df['age_'].isnull()].index, inplace=True)
    df.drop(df[df['MaritalStatus']=='Unknown'].index, inplace=True)
    
    get_ipython().magic('matplotlib inline')
    hist = df['age_'].hist()
    df['age_'].mean()
    
    age_bins = [0, 22, 100]
    age_groups = pd.cut(df['age_'], bins=age_bins)
        
    df.groupby('Ethnic_Code_Text').count()
    
    df['CustodyStatus_'] = -1
    mask = df['CustodyStatus']=='Jail Inmate'
    df.loc[mask, 'CustodyStatus_'] = 0
    df.loc[~mask, 'CustodyStatus_'] = 1
    
    df['LegalStatus_'] = -1
    mask = df['LegalStatus']=='Post Sentence'
    df.loc[mask, 'LegalStatus_'] = 0
    df.loc[~mask, 'LegalStatus_'] = 1
    
    df['RecSupervisionLevel_'] = -1
    mask = df['RecSupervisionLevel'].isin((1, 2))
    df.loc[mask, 'RecSupervisionLevel_'] = 0
    df.loc[~mask, 'RecSupervisionLevel_'] = 1
    
    df['Ethnic_Code_Text_'] = -1
    mask = df['Ethnic_Code_Text']=='Caucasian'
    df.loc[mask, 'Ethnic_Code_Text_'] = 0
    df.loc[~mask, 'Ethnic_Code_Text_'] = 1
    
    df['MaritalStatus_'] = -1
    mask = df['MaritalStatus'].isin(('Married', 'Significant Other'))
    df.loc[mask, 'MaritalStatus_'] = 0
    df.loc[~mask, 'MaritalStatus_'] = 1
    
    age_bins = [0, 30, 100]
    age_groups = pd.cut(df['age_'], bins=age_bins)
    df['Age'] = age_groups
    num_groups = len(df['Age'].cat.categories)
    df['Age'] = df['Age'].cat.rename_categories(range(num_groups))
    get_ipython().magic('matplotlib inline')
    hist = df['Age'].hist()
    
    df['ScoreText_'] = -1
    mask = df['ScoreText']=='High'
    df.loc[mask, 'ScoreText_'] = 0
    df.loc[~mask, 'ScoreText_'] = 1
    
    df['Sex_'] = -1
    mask = df['Sex_Code_Text']=='Male'
    df.loc[mask, 'Sex_'] = 0
    df.loc[~mask, 'Sex_'] = 1    
    
    df.drop(['DisplayText', 'Sex_Code_Text', 'ScoreText','Agency_Text', 'AssessmentType', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason', 'Language'], axis=1, inplace=True)
    df.drop(['DateOfBirth', 'age_', 'MaritalStatus', 'Ethnic_Code_Text', 'RecSupervisionLevel','CustodyStatus','LegalStatus', 'Scale_ID'], axis=1, inplace=True)

    binarized_path = os.path.join(repor_root,'data','compass_binerized.csv') 
    df.to_csv(binarized_path, index=False)

    labels = df["ScoreText_"]
    eg_from_df = df["Ethnic_Code_Text"].to_numpy()
    features = df.drop(columns="ScoreText_", "Ethnic_Code_Text")
    return features, labels, eg_from_df

# "ScoreText_"

def load_compas_data_pcategorizing(data_path):

    df = pd.read_csv(data_path, 
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

    age = (datetime.datetime.now() - df.DateOfBirth).astype('timedelta64[Y]')
    age = age.astype('int')
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
  
    categorized_path = os.path.join(repor_root,'data','compass_categorized.csv') 
    df.to_csv(categorized_path, index=False)

    labels = df["ScoreText_"]
    eg_from_df = df["Ethnic_Code_Text"].to_numpy()
    features = df.drop(columns="ScoreText_", "Ethnic_Code_Text")
    return features, labels, eg_from_df

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
def lr_schedule(epoch):
    """Defines the learning rate schedule."""
    if epoch < 20:
        return 1e-4
    else:
        return 1e-5

def get_model_z(X,s,n_z,model_name,epochs=20,verbose=1,var_reg=0.0):
    """
    Defines and trains a clustering model. 
    """
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
        Dense(n_z, activation='softmax', name='target_layer'),
        Dense(s.shape[1], activation='linear', use_bias=False,kernel_initializer=RandomUniform(minval=0, maxval=1),kernel_constraint=ClipConstraint(0, 1), kernel_regularizer=VarianceRegularizer(factor=var_reg)), 
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

def pzx(X,best_model,arg_max=True):
    """
    Predict cluster assignments
    """
    softmax_output_model = Model(inputs=best_model.input, outputs=best_model.layers[-2].output)
    p = softmax_output_model.predict(X)
    if(arg_max):
        p = np.argmax(p,axis=1)
    return p

def set_seed(seed_num): 
    # Set random seed for reproducibility
    seed = seed_num
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ===========================
# Census Data Set
# ===========================

def run_compas(data_path, output_csv_name, output_model_results, num_clusters=2, num_epochs=40, model_path='best_census_model_inc.h5', num_var_reg=0, seed_num=0):
    
    set_seed(seed_num)

    features, labels, eg_from_df = load_compas_data_pcategorizing(data_path)
    s1 = to_categorical(labels)
    best_model = get_model_z(features, s1, num_clusters, model_path, num_epochs, num_var_reg)

    p1_tr = pzx(features, best_model, arg_max=True)

    
    cluster_eg_df = pd.DataFrame({
        "cluster":p1_tr,
        "ethnic_group":states_from_df, 
        }
        )

    # Save the DataFrame to a CSV file
    output_folder = os.path.join(repo_root, "eg_retrieved_data")
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, output_csv_name)
    cluster_state_df.to_csv(output_csv_path, index=False)

    # Group by 'states' and 'cluster', then count the occurrences of each cluster for each state
    cluster_counts = cluster_eg_df.groupby(["ethnic_group", "cluster"]).size().reset_index(name='count')
    
    # Calculate the total count for each state
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


# def run_census_cosine(census_data_csv_path, cosine_census_path_name): 
#     """
#     census_data_csv_path : should be the one after run_census - will be in the census_retrieved_data folder
#     output_cosine_census_results_path : just the name of the jaccard plot - should be .png and indicate data processor
#     """
#     census_data = pd.read_csv(census_data_csv_path) 
    
#     # Rename column if necessary
#     census_data = census_data.rename(columns={"p1_tr": "cluster"})
    
#     # Create a binary matrix for states and clusters
#     state_cluster_matrix = pd.crosstab(census_data['states'], census_data['cluster'])
    
#     # Convert to a numpy array to calculate distances
#     state_cluster_matrix_array = state_cluster_matrix.values
    
#     # Calculate the pairwise Cosine similarity between states
#     cosine_sim_matrix = cosine_similarity(state_cluster_matrix_array)
    
#     # Convert the result to a DataFrame for better readability
#     cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=state_cluster_matrix.index, columns=state_cluster_matrix.index)
    
#     # Plot the cosine similarity matrix as a heatmap
#     plt.figure(figsize=(15, 8))
#     sns.heatmap(cosine_sim_df, annot=False, cmap="Blues", cbar=True, xticklabels=True, yticklabels=True)
#     plt.title('State Similarity Based on Clusters (Cosine Similarity)')
    
#     # Save the plot as a PNG file
#     output_folder = os.path.join(repo_root, "census_image_results")
#     os.makedirs(output_folder, exist_ok=True)
#     output_file = os.path.join(output_folder, cosine_census_path_name)
#     plt.savefig(output_file)


# def load_compas_data(data_path, seed_num=0):
#    """
#    Loading compas data
#    """
#     set_seed(seed_num)
#     data = pd.read_csv(data_path)

#     # Remove unique and duplicate features
#     data = data.loc[:, ~data.columns.duplicated()]
#     unique_threshold = len(data) * 0.95  # 95% unique threshold
#     data = data.loc[:, data.nunique() < unique_threshold]

#     # Remove rows with low-frequency counts in categorical columns
#     for col in data.select_dtypes(include='object').columns:
#         value_counts = data[col].value_counts()
#         valid_values = value_counts[value_counts >= low_freq_threshold].index
#         data = data[data[col].isin(valid_values)]

#     ethnic_groups = data["Ethnic_Code_Text"].unique()

#     ## GRAB ONLY CERTAIN AMOUNT FROM EACH ETHNIC GROUP
#     ## FOR EACH Ethinic Group, APPLY MIN, MAX SCALER (both)
#     min_row_count = data.groupby("Ethnic_Code_Text").size().min()

#     # Sample the minimum number of rows for each state
#     balanced_data = pd.DataFrame()
#     for eg in ethnic_groups:
#         eg_data = data[data['Ethnic_Code_Text'] == eg]
#         if len(eg_data) >= min_row_count:
#             sampled_data = eg_data.sample(n=min_row_count)
#             balanced_data = pd.concat([balanced_data, sampled_data], ignore_index=True)

#     # Encode state labels
#     label_encoder = LabelEncoder()
#     eg_from_df = balanced_data["Ethnic_Code_Text"].to_numpy()
#     balanced_data['eg_encoded'] = label_encoder.fit_transform(balanced_data['Ethnic_Code_Text'])
#     labels = balanced_data["eg_encoded"].to_numpy()
    
#     features = balanced_data.drop(columns=["Ethnic_Code_Text", "eg_encoded"]).to_numpy()
#     standard_scaler = StandardScaler()
#     features_standardized = standard_scaler.fit_transform(features)

#     minmax_scaler = MinMaxScaler()
#     features_scaled = minmax_scaler.fit_transform(features_standardized)

#     return features_scaled, labels, eg_from_df
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    