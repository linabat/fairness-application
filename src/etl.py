from ucimlrepo import fetch_ucirepo 

import pandas as pd 
import numpy as np
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


def retrieve_covid_data(covid_fp, replace_num, test_data=False): 
    """
    This function is used to retrieve the covid dataset
    """
    if test_data == False:
        covid = pd.read_csv(covid_fp)
        # Cleaning up column names
        covid.columns = covid.columns.str.strip().str.lower()
    
        # Creating boolean column which is what will be predicted
        covid["died_bool"] = covid["date_died"] != "9999-99-99"
    
        # Replacing all 98 values with 97 so there is only one number that indicates whether
        # the value is missing
        covid.replace(replace_num, 97, inplace=True)
        
        covid.drop(columns=["clasiffication_final", "date_died"], inplace=True)
        
        return covid 
    else: 
        test = pd.read_csv(covid_fp)
        # Cleaning up column names
        test.columns = test.columns.str.strip().str.lower()
        return test

### GMM
def gmm_adults(gmm_adult_ts): 
    """
    This function prints out a classification report for the Gaussian Mixture Model that
    is used to identify 2 clusters to predict whether someone will have an income greater than 
    or less than 50,000
    """
    # Retrieving data for model 
    X,y = retrieve_adult_data()
    X = pd.get_dummies(X, drop_first=True)
    
    # standardizing features 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = gmm_adult_ts)

    gmm = GaussianMixture(n_components = 2)

    gmm.fit(X_train)

    y_pred = gmm.predict(X_test)

    mapped_y_pred = [0 if label == y_test.mode()[0] else 1 for label in y_pred]
    
    output_model_folder = os.path.join(repo_root, "gmm_kmeans_results")
    os.makedirs(output_model_folder, exist_ok=True)
    output_txt_path = os.path.join(output_model_folder, "gmm_adults_results.txt")
    with open(output_txt_path, "w") as file:
        # Proportions for each cluster
        file.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
        file.write(f"Classification Report: {classification_report(y_test, y_pred)}\n")

def gmm_covid(covid_fp, replace_num, gmm_covid_ts,test_data=False): 
    """
    This function outputs a classification report for the the Gaussian Mixture model for 
    covid dataset - am only looking at it's ability to identify 2 groups. 
    """
    covid = retrieve_covid_data(covid_fp, replace_num)

    class_0 = covid[covid["died_bool"] == False]
    class_1 = covid[covid["died_bool"] == True]
    class_1_count = class_1.shape[0]

    class_0_under = class_0.sample(class_1_count)

    # Equal numbers of died and not died in this datasets
    covid_under = pd.concat([class_0_under, class_1], axis=0)

    # Separate the target variable
    y = covid_under["died_bool"]
    X = covid_under.drop(columns=["died_bool"])

    # Convert categorical variables to dummy/indicator variables
    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified train-test split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=gmm_covid_ts, random_state=42
    )

    # Fit the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=2)
    gmm.fit(X_train)

    # Predict on test data
    y_pred = gmm.predict(X_test)

    # Map predictions to 0 or 1 based on the most common label in y_test
    mapped_y_pred = [0 if label == y_test[0] else 1 for label in y_pred]

    output_model_folder = os.path.join(repo_root, "gmm_kmeans_results")
    os.makedirs(output_model_folder, exist_ok=True)
    output_txt_path = os.path.join(output_model_folder, "gmm_covid_results.txt")
    with open(output_txt_path, "w") as file:
        # Proportions for each cluster
        file.write(f"Accuracy: {accuracy_score(y_test, mapped_y_pred)}\n")
        file.write(f"Classification Report: {classification_report(y_test, mapped_y_pred)}\n")
    
    return X_scaled, y, gmm

def plot_pca_gmm_covid(covid_fp, replace_num, gmm_covid_ts, test_data=False):
    """
    This function is used to plot compare the two group that the GMM identifies 
    to the 2 original groups.
    """
    
    X_scaled, y, gmm = gmm_covid(covid_fp, replace_num, gmm_covid_ts)
    # Perform PCA to reduce the dataset to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Predict clusters using GMM
    y_cluster = gmm.predict(X_scaled)

    # Create a DataFrame with PCA results, GMM clusters, and original class labels
    pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
    pca_df['GMM Cluster'] = y_cluster
    pca_df['Original Class'] = y # Assuming `y_sample` is the original target label

    # Plot side-by-side comparison of GMM clusters and original classes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot GMM Clusters
    sns.scatterplot(x='PCA1', y='PCA2', hue='GMM Cluster', data=pca_df, palette='Set1', ax=ax1, alpha=0.7)
    ax1.set_title('GMM Clusters')
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.legend(title='GMM Cluster')

    # Plot Original Classes
    sns.scatterplot(x='PCA1', y='PCA2', hue='Original Class', data=pca_df, palette='Set2', ax=ax2, alpha=0.7)
    ax2.set_title('Original Classes')
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    ax2.legend(title='Original Class')

    plt.tight_layout()
    # Save the plot as a PNG file
    output_folder = os.path.join(repo_root, "gmm_kmeans_results")
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, "gmm_covid_pca")

    plt.savefig(output_file_path, dpi=300)

def kmeans_adults(): 
    X, y = retrieve_adult_data()
    data = pd.concat([X, y], axis=1)
    
    # Encode categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    data[categorical_cols] = data[categorical_cols].apply(LabelEncoder().fit_transform)
    
    # Standardize numerical features
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Apply k-means clustering
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=0)
    data['cluster'] = kmeans.fit_predict(data[numeric_cols])
    
    # Calculate silhouette score
    score = silhouette_score(data[numeric_cols], data['cluster'])
    print(f'Silhouette Score for {k} clusters: {score}')
    
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(data[numeric_cols])  # Use only numeric features
    
    # Plot the clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=data['cluster'], palette='viridis', s=50)

    # Save the plot as a PNG file
    output_folder = "gmm_kmeans_results"
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, "kmeans_adults.png")
    plt.savefig(output_file_path, dpi=300)

def retrieve_adult_data():
    """
    This function is used to retrieve UCI's adult dataset
    """
    # Fetch dataset 
    adult = fetch_openml(name='adult', version=2, as_frame=True)

    # Data (as pandas dataframes) 
    X = adult.data
    y = adult.target
    
    # Cleaning target values
    y.replace("<=50K.", "<=50K", inplace=True)
    y.replace(">50K.", ">50K", inplace=True)
    
    # Group the original dataset for how it was 
    full_data = pd.concat([X, y], axis=1)

    # Breaking up the groups so we can do undersampling for the "greater than" group
    less_than = full_data[full_data["class"] == "<=50K"]
    greater_than = full_data[full_data["class"] == ">50K"]

    # Conducting undersampling here 
    greater_than_count = greater_than.shape[0]
    less_than_under = less_than.sample(greater_than_count)

    under_sampled_data = pd.concat([greater_than, less_than_under], axis=0)
    under_sampled_data["lower_income_bool"] = under_sampled_data["class"] == "<=50K"

    y = under_sampled_data["lower_income_bool"]
    X = under_sampled_data.drop(columns=["class", "lower_income_bool"])
    
    return X, y



# Image Pre-Processing 
# ===========================
# Feature Extraction
# ===========================
def create_feature_extractor():
    """
    Model that extracts features from images using a pre-trained ResNet50
    """
    base_model = ResNet50(weights='imagenet', include_top=False)
    extractor = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    return extractor

def process_images_in_batches(dataset_path, metadata_df, batch_size=32):
    """
    Process images in batches to extract features while minimizing memory usage
    """
    extractor = create_feature_extractor()
    total_images = len(metadata_df)
    features = np.zeros((total_images, 2048)) # Resizing images to 224x224 pixels to match 
    # ResNet50's input size
    
    for i in tqdm(range(0, total_images, batch_size)):
        batch_files = metadata_df['img_filename'].iloc[i:i+batch_size]
        batch_images = []
        
        for img_path in batch_files:
            img = image.load_img(os.path.join(dataset_path, img_path), 
                               target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            batch_images.append(img_array)
        
        batch_images = np.vstack(batch_images)
        batch_images = preprocess_input(batch_images)
        
        batch_features = extractor.predict(batch_images, verbose=0)
        features[i:i+len(batch_files)] = batch_features
        
        del batch_images
        del batch_features
    
    return features # a 2D numpy array where each row represents the extracted features for an image
    

# Census Pre-Processing

def load_census_data(data_processor, seed_num=0):
    """
    Load and process census data.
    @param data_processor: ACSIncome, ACSEmployment, ACSPublicCoverage 
    """
    set_seed(seed_num)

    states = ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA',
        'MA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NH','NJ','NM','NY','NV','OH','OK','OR','PA','PR',
        'RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY']
    
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    all_data = pd.DataFrame()
    state_row_counts = {}

    for state in states: 
        try:
            state_data = data_source.get_data(states=[state], download=True)     
            state_features, _, _ = data_processor.df_to_numpy(state_data)
            state_features_df = pd.DataFrame(state_features)
            state_features_df['ST'] = state
            state_row_counts[state] = state_features_df.shape[0]
            all_data = pd.concat([all_data, state_features_df], ignore_index=True)
        except Exception as e: 
            continue

    ## GRAB ONLY CERTAIN AMOUNT FROM EACH STATE
    ## FOR EACH STATE, APPLY MIN, MAX SCALER (both)
    
    min_row_count = min(state_row_counts.values())
    # Sample the minimum number of rows for each state
    balanced_data = pd.DataFrame()
    for state in states:
        state_data = all_data[all_data['ST'] == state]
        if len(state_data) >= min_row_count:
            sampled_data = state_data.sample(n=min_row_count)
            balanced_data = pd.concat([balanced_data, sampled_data], ignore_index=True)

    # Encode state labels
    label_encoder = LabelEncoder()
    states_from_df = balanced_data["ST"].to_numpy()
    balanced_data['ST_encoded'] = label_encoder.fit_transform(balanced_data['ST'])
    labels = balanced_data["ST_encoded"].to_numpy()
    
    features = balanced_data.drop(columns=["ST", "ST_encoded"]).to_numpy()
    standard_scaler = StandardScaler()
    features_standardized = standard_scaler.fit_transform(features)

    minmax_scaler = MinMaxScaler()
    features_scaled = minmax_scaler.fit_transform(features_standardized)

    return features_scaled, labels, states_from_df

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

def visualization_images(file_paths, p, y, p_value, y_value, dataset_path):
    """
    Visualizes a 2x2 grid of images based on specified filtering conditions.
    """
    selected_indices = np.random.choice(len(file_paths[(p==p_value)&(np.squeeze(y)==y_value)]), 4, replace=False)
    selected_file_paths = file_paths[(p==p_value)&(np.squeeze(y)==y_value)][selected_indices]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    for ax, file_path in zip(axes.flatten(), selected_file_paths):
        file = os.path.join(dataset_path, file_path)
        img = Image.open(file)
        ax.imshow(img)
        ax.axis('off')  
    
    plt.tight_layout()

    # Save the plot as a PNG file
    output_folder = os.path.join(repo_root, "waterbirds_image_results")
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, f"{p_value}_{y_value}.png")

    plt.savefig(output_file_path, dpi=300)
    
def download_wb_data(tar_file_path): 
    tar_file_path = os.path.join(repo_root, tar_file_path)
    extract_path = os.path.join(repo_root, "waterbirds_data")
    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_path, exist_ok=True)

    # Open and extract the .tar.gz file
    with tarfile.open(tar_file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted contents to: {extract_path}")
        
def retrieve_wb_features(): 
    # Know this path cause this is where images were saved from download_waterbirds_data function
    dataset_path = os.path.join(repo_root, "waterbirds_data/waterbird_complete95_forest2water2")
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_file)

    features = process_images_in_batches(dataset_path, metadata_df)

    output_folder = os.path.join(repo_root, "waterbirds_features")
    os.makedirs(output_folder, exist_ok=True)

    output_file_path = os.path.join(output_folder, "features.npy")
    
    np.save(output_file_path, features)
# ===========================
# Waterbirds Run
# ===========================
def run_waterbirds(output_csv_name, output_model_results, num_clusters=2, num_epochs=60, model_path='best_wb_model_inc.h5', num_var_reg=0, seed_num=0): 
    set_seed(seed_num)

    dataset_path = dataset_path = os.path.join(repo_root, "waterbirds_data/waterbird_complete95_forest2water2")
    features_path = os.path.join(repo_root, "waterbirds_features/features.npy")
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_file)
    y = metadata_df['y']

    s = metadata_df['split'].values 
    s=(s+1)//2

    features = np.load(features_path)
    file_paths = metadata_df['img_filename'].values

    # Make s1 as one hot of s
    s1 = to_categorical(s)
        
    best_model = get_model_z(features, s1, num_clusters, model_path, epochs=num_epochs, var_reg=num_var_reg)
    
    # for function pzx - want to save these each to a column 
    p1_fl = pzx(features, best_model, arg_max=False)
    p1_tr = pzx(features, best_model, arg_max=True)

    file_paths = metadata_df['img_filename'].values
    
    y = metadata_df['y']
    place = metadata_df['place'] # check if y and place match and output into a txt file

    p1_fl_df = pd.DataFrame(
        p1_fl,
        columns=['p1_fl_cluster_0', 'p1_fl_cluster_1']
    )

    p1_tr_df = pd.DataFrame(
        p1_tr,
        columns=['cluster']
    )

    # Combine everything into a DataFrame
    p_y_place_df = pd.concat(
        [p1_fl_df, p1_tr_df,
            pd.DataFrame({
                "y": y,
                "place": place
            })
        ],
        axis=1
    )

    # Save the DataFrame to a CSV file
    output_folder = os.path.join(repo_root, "wb_retrieved_data")
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, output_csv_name)
    p_y_place_df.to_csv(output_csv_path, index=False)


    ## EVALUATION OF MODEL OUTPUT

    # Creating a new column where the condition y == place is checked
    p_y_place_df['y_equals_place'] = p_y_place_df['y'] == p_y_place_df['place']
    
    # Group by 'cluster' and calculate the proportion
    proportions = (
        p_y_place_df.groupby('cluster')['y_equals_place']
        .mean()  # Mean of True/False (1/0) gives the proportion
        .reset_index(name='proportion_y_equals_place')
    )

    output_model_folder = os.path.join(repo_root, "model_results")
    os.makedirs(output_model_folder, exist_ok=True)
    output_txt_path = os.path.join(output_model_folder, output_model_results)
    with open(output_txt_path, "w") as file:
        # Proportions for each cluster
        file.write("Proportion Accuracy Where Y=Place:\n")
        for _, row in proportions.iterrows():
            file.write(f"Cluster {row['cluster']}: {row['proportion_y_equals_place']:.4f}\n")


    # Doing all the combination to see what it should be
    visualization_images(file_paths, p1_tr, y, 1, 0, dataset_path)
    visualization_images(file_paths, p1_tr, y,1, 1, dataset_path)
    visualization_images(file_paths, p1_tr, y, 0, 1, dataset_path)
    visualization_images(file_paths, p1_tr, y, 0, 0, dataset_path)

# ===========================
# Census Data Set
# ===========================

def run_census(data_processor_type, output_csv_name, output_model_results, num_clusters=4, num_epochs=60, model_path='best_census_model_inc.h5', num_var_reg=0, seed_num=0):
    
    set_seed(seed_num)

    if data_processor_type == "income": 
        data_processor = ACSIncome

    elif data_processor_type == "employment": 
        data_processor = ACSEmployment

    elif data_processor_type == "public_coverage": 
        data_processor =  ACSPublicCoverage

    else: 
        raise ValueError(f"Invalid data_processor_type '{data_processor_type}'. Must be one of: 'income', 'employment', or 'public_coverage'.")

    
    features, labels, states_from_df = load_census_data(data_processor)
    s1 = to_categorical(labels)
    best_model = get_model_z(features, s1, 4, model_path, epochs=40, var_reg=0)

    p1_tr = pzx(features, best_model, arg_max=True)

    
    cluster_state_df = pd.DataFrame({
        "cluster":p1_tr,
        "states":states_from_df, 
        "type": data_processor_type
        }
        )

    # Save the DataFrame to a CSV file
    output_folder = os.path.join(repo_root, "census_retrieved_data")
    os.makedirs(output_folder, exist_ok=True)
    output_csv_path = os.path.join(output_folder, output_csv_name)
    cluster_state_df.to_csv(output_csv_path, index=False)

    # Group by 'states' and 'cluster', then count the occurrences of each cluster for each state
    cluster_counts = cluster_state_df.groupby(["states", "cluster"]).size().reset_index(name='count')
    
    # Calculate the total count for each state
    state_totals = cluster_state_df.groupby("states")["cluster"].count().reset_index(name='total')
    
    # Merge the counts with the total counts per state
    merged = pd.merge(cluster_counts, state_totals, on="states")
    
    # Calculate the proportion for each cluster within each state
    merged['proportion'] = merged['count'] / merged['total']
    result = merged.loc[merged.groupby("states")["proportion"].idxmax()]
    cluster_groups = result.groupby("cluster")["states"].apply(list)



    output_model_folder = os.path.join(repo_root, "model_results")
    os.makedirs(output_model_folder, exist_ok=True)
    output_txt_path = os.path.join(output_model_folder, output_model_results)
    with open(output_txt_path, "w") as file:
        file.write("State - Cluster Mapping Based on Highest Proportion:\n")
        file.write("=" * 50 + "\n\n")
        for cluster, states in cluster_groups.items():
            state_list = ", ".join(states)
            file.write(f"Cluster {cluster}: = [{state_list}]\n")


def run_census_cosine(census_data_csv_path, cosine_census_path_name): 
    """
    census_data_csv_path : should be the one after run_census - will be in the census_retrieved_data folder
    output_cosine_census_results_path : just the name of the jaccard plot - should be .png and indicate data processor
    """
    census_data = pd.read_csv(census_data_csv_path) 
    
    # Rename column if necessary
    census_data = census_data.rename(columns={"p1_tr": "cluster"})
    
    # Create a binary matrix for states and clusters
    state_cluster_matrix = pd.crosstab(census_data['states'], census_data['cluster'])
    
    # Convert to a numpy array to calculate distances
    state_cluster_matrix_array = state_cluster_matrix.values
    
    # Calculate the pairwise Cosine similarity between states
    cosine_sim_matrix = cosine_similarity(state_cluster_matrix_array)
    
    # Convert the result to a DataFrame for better readability
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=state_cluster_matrix.index, columns=state_cluster_matrix.index)
    
    # Plot the cosine similarity matrix as a heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(cosine_sim_df, annot=False, cmap="Blues", cbar=True, xticklabels=True, yticklabels=True)
    plt.title('State Similarity Based on Clusters (Cosine Similarity)')
    
    # Save the plot as a PNG file
    output_folder = os.path.join(repo_root, "census_image_results")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, cosine_census_path_name)
    plt.savefig(output_file)

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    