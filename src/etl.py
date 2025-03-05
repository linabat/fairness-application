import pandas as pd 
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import datetime

from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import train_test_split

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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tqdm import tqdm
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, Flatten, 
    MaxPooling2D, BatchNormalization, Dropout, Concatenate
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler
)

from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.models import Sequential, Model
from keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import pairwise_distances
from PIL import Image, ImageDraw, ImageFont

#### Cross validation  
from sklearn.model_selection import StratifiedKFold
from itertools import product
from sklearn.base import BaseEstimator, ClassifierMixin

# This will be used when saving the files
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -------------------------------
# Custom Gradient Reversal Layer
# -------------------------------
@tf.custom_gradient
def grad_reverse(x, lambda_):
    def grad(dy):
        return -lambda_ * dy, None # reverses direction of gradient 
    return x, grad

# custom Keras layer
"""
Layer is used to ensure that the feature representation are independent of a sensitive attribute
- feature extract learns normally in the forward pass
- reversing gradients of classifier that tries to predict the sensitive attribute during backpropagation -- stops feature extractor from encoding sensitive information
"""
class GradientReversalLayer(tf.keras.layers.Layer): 
    def __init__(self, lambda_=1.0, **kwargs):
        super(GradientReversalLayer, self).__init__(**kwargs)
        self.lambda_ = lambda_ # strength of gradient reversal
    def call(self, x):
        return grad_reverse(x, self.lambda_)

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    tf.random.set_seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)

# -------------------------------
# Adversarial Debiasing Model
# -------------------------------
def build_adversarial_model(input_dim, lambda_adv=1.0):
    """
    Build an adversarial debiasing model that learns pseudo‑labels Y' from X.

    Architecture:
      - Main branch (encoder): from X, several dense layers produce a latent pseudo‑label pseudo_Y (via sigmoid).
      - Adversary branch: pseudo_Y is passed through a Gradient Reversal Layer and then dense layers predict S.
      - Decoder branch: concatenates pseudo_Y and the one-hot sensitive attribute S to predict the observed label Y.

    Losses:
      - For the main branch, binary crossentropy between observed Y and pseudo_Y (and Y_pred).
      - For the adversary branch, categorical crossentropy to predict S.

    Returns a compiled Keras model that takes inputs X and S (one-hot encoded) and outputs:
      [pseudo_Y, S_pred, Y_pred].
    """
    X_input = tf.keras.Input(shape=(input_dim,), name="X")
    S_input = tf.keras.Input(shape=(2,), name="S")  # one-hot encoded S

    # Main branch: Encoder for pseudo-label.
    """
    """
    h = Dense(64, activation='relu')(X_input)
    h = BatchNormalization()(h)
    h = Dense(32, activation='relu')(h)
    h = BatchNormalization()(h)
    pseudo_Y = Dense(1, activation='sigmoid', name="pseudo_Y")(h) ## outputs  probability value for pseudo_Y between 0,1

    # Adversary branch: from pseudo_Y, with GRL.
    """
    This is to prevent psuedo_Y from containing information about S
    - adversary will try to predict S from pseudo_Y (fair label)...if it can accurately predict S, then Y' still encodes information about S (don't want this) 
    - use the gradient reversal layer to prevent this from happening
    """
    grl = GradientReversalLayer(lambda_=lambda_adv)(pseudo_Y)
    a = Dense(32, activation='relu')(grl)
    a = BatchNormalization()(a)
    S_pred = Dense(2, activation='softmax', name="S_pred")(a)

    # Decoder branch: combine pseudo_Y and S to predict observed Y.
    """
    Y depepends on both Y' and S 
    -- predict the final observed label Y using both psuedo_Y and S
    -- Y may still depend on S, that is why it's being used here 
    -- decoder ensures Y_final is accurate, while psuedo_Y is not directly influenced by S 
    -- psuedo_Y removes unfair dependencies on S...however S might still contain legit info needed to predict Y accurately 
    -- IMPORTANT - THIS STEP ALLOWS FAIR DEPENDENCIES WHILE ELIMINATING UNFAIR ONES
    -- structure how S influences Y, without letting hidden biases leak through 
    """
    concat = Concatenate()([pseudo_Y, S_input])
    d = Dense(16, activation='relu')(concat)
    d = BatchNormalization()(d)
    Y_pred = Dense(1, activation='sigmoid', name="Y_pred")(d)

    model = tf.keras.Model(inputs=[X_input, S_input],
                           outputs=[pseudo_Y, S_pred, Y_pred])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss={"pseudo_Y": "binary_crossentropy",
                        "S_pred": "categorical_crossentropy",
                        "Y_pred": "binary_crossentropy"},
                  loss_weights={"pseudo_Y": 1.0, "S_pred": lambda_adv, "Y_pred": 1.0},
                  metrics={"pseudo_Y": "accuracy",
                           "S_pred": "accuracy",
                           "Y_pred": "accuracy"}) # Y_pred is the best estimate of Y accounting for fair dependencies 
    return model

# -------------------------------
# Manual Fairness Metrics
# -------------------------------
def compute_fairness_metrics_manual(y_true, y_pred, sensitive_features):
    """
    Compute fairness metrics manually.
    y_true: binary ground-truth labels (1-D numpy array).
    y_pred: continuous scores (will be thresholded at 0.5).
    sensitive_features: 1-D numpy array (0 or 1).

    Returns a dictionary with:
      - Demographic parity difference (absolute difference in positive rates).
      - Equalized odds difference (average difference in TPR and FPR).
      - Selection rates per group.
      - Group-wise accuracy.
    """
    y_pred_bin = (y_pred > 0.5).astype(int) # y_pred is continuous value, so converting it to binary 
    groups = np.unique(sensitive_features)

    # Demographic parity 
    """
    All groups (from sensitive feature) should receive positive predictions at the same rate
    P(Y_hat = 1|S=0) = P(Y_hat=1|S=1)
    """

    # For each group in the sensitive feature, find the demographic parity and compute the difference (based on the formula in above comment)
    pos_rates = {}
    for g in groups: 
        pos_rates[g] = np.mean(y_pred_bin[sensitive_features == g])
    dp_diff = abs(pos_rates[0] - pos_rates[1]) ## this line assumes that there are only 2 groups, 0 and 1 
    # dp_diff > 0, then demographic parity isn't fair 

    # Equalized odds
    """
    Ensuring the different groups in the sensitive feature similar TPR and FPR rates -- this is so that the model isn't discriminating in error types
    """
    metrics = {}
    for g in groups:
        mask = (sensitive_features == g)
        y_true_g = y_true[mask]
        y_pred_g = y_pred_bin[mask]
        tpr = np.sum((y_pred_g == 1) & (y_true_g == 1)) / (np.sum(y_true_g == 1) + 1e-8) # True Positive Rate
        fpr = np.sum((y_pred_g == 1) & (y_true_g == 0)) / (np.sum(y_true_g == 0) + 1e-8) # False Positive Rate
        metrics[g] = (tpr, fpr)
    eo_diff = (abs(metrics[0][0] - metrics[1][0]) + abs(metrics[0][1] - metrics[1][1]))/2 # taking average of two error types

    # Selection rate per group.
    """
    proportion of samples predicted as positive for each group -- a a group has a higher selection rate, the model may favor that group unfairly
    """
    sel_rate = {}
    for g in groups:
        sel_rate[g] = pos_rates[g]

    # Group-wise accuracy.
    """
    for each group in the sensitive feature, compute the accuracy of the model (to ensure that it's perfoming consistently across groups)
    """
    group_acc = {}
    for g in groups:
        mask = (sensitive_features == g)
        group_acc[g] = accuracy_score(y_true[mask], y_pred_bin[mask])

    return {
        "demographic_parity_difference": dp_diff,
        "equalized_odds_difference": eo_diff,
        "selection_rate": sel_rate,
        "group_accuracy": group_acc
    }

# -------------------------------
# Plotting Function
# -------------------------------
def plot_comparison(metrics_baseline, metrics_fair, plot_file_path_4, plot_file_path_2):
    """
    Generates a comparison plot and a table displaying numerical values of evaluation metrics.
    """
    models = ['Baseline', 'Fair Model']
    accs = [metrics_baseline['accuracy'], metrics_fair['accuracy']]
    dp_diff = [metrics_baseline["demographic_parity_difference"], metrics_fair["demographic_parity_difference"]]
    aucs = [metrics_baseline['auc'], metrics_fair['auc']]
    eo_diff = [metrics_baseline["equalized_odds_difference"], metrics_fair["equalized_odds_difference"]]


    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Create secondary y-axis
    
    width = 0.4  # Bar width
    gap = 0.015   # Space between bars
    x_indexes = np.arange(len(models))

    # color blind friendly options
    color_acc = "#006CD1"  # blue (Accuracy)
    color_dp = "#994F00"   # brown (Demographic Parity)

    bars1 = ax1.bar(x_indexes - (width/2 + gap), accs, width, color=color_acc, label="Accuracy")
    bars2 = ax2.bar(x_indexes + (width/2 + gap), dp_diff, width, color=color_dp, label="Demographic \n Parity Difference")

    # value is included at the top of the bar
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.2f}', 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # remove all axis lines (keeping it minimal
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    ax1.yaxis.set_visible(False)
    ax2.yaxis.set_visible(False)

    # Labels & Titles
    ax1.set_xticks(x_indexes)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_title("Accuracy & Fairness Comparison", fontsize=14)

    # Legend (smaller and in top right corner)
    ax1.legend(handles=[bars1, bars2], loc='upper right', fontsize=10)

    # Save plot
    plt.tight_layout()
    plt.savefig(plot_file_path_2, bbox_inches="tight")
    plt.close()

    ## plot with all 4 metrics
    
    # Creating a 2x2 grid of bar charts comparing metrics
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].bar(models, aucs, color=['blue', 'green'])
    axs[0, 0].set_title('AUC')
    axs[0, 0].set_ylim([0, 1])

    axs[0, 1].bar(models, accs, color=['blue', 'green'])
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_ylim([0, 1])

    axs[1, 0].bar(models, dp_diff, color=['orange', 'purple'])
    axs[1, 0].set_title('Demographic Parity Difference')

    axs[1, 1].bar(models, eo_diff, color=['orange', 'purple'])
    axs[1, 1].set_title('Equalized Odds Difference')

    # plt.suptitle("Comparison: Baseline (X → Y) vs. Fair (X → Y') Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(plot_file_path_4, bbox_inches="tight")
    plt.close()

# -------------------------------
# Main Function: Comparison and Visualization
# -------------------------------
def main_binary(data_url, dataset_name, lambda_adv=1.0, epochs=64, batch_size=128, output_dir='model_results'):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # creating folder for output results
    output_path = os.path.join(repo_root, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    log_file_path = os.path.join(output_path, f'{dataset_name}_results_log.txt')
    plot_file_path_4 = os.path.join(output_path, f'4_{dataset_name}_comparison_plot.png')
    plot_file_path_2 = os.path.join(output_path, f'2_{dataset_name}_comparison_plot.png')


    def log(message):
        with open(log_file_path, 'a') as f: 
            f.write(message + '\n')
    
    open(log_file_path, 'w').close() 

    set_seed(42)
                     
    if dataset_name == "compas": 
        X, Y_obs, S = load_and_preprocess_compas_data_binary(data_url) ##  S is binary

    elif dataset_name == "german":
        X, Y_obs, S = load_and_preprocess_german_data(data_url) ##  S is binary


    elif dataset_name == "adult":
        X, Y_obs, S = load_and_preprocess_adult_data(data_url) ##  S is binary

    else:
        log("Invalid dataset_name")
        return 
    
    log(f"Loading and preprocessing {dataset_name} data...")
    X_train, X_test, Y_train_obs, Y_test_obs, S_train, S_test = train_test_split(
        X, Y_obs, S, test_size=0.2, random_state=42
    )

    if dataset_name == "compas":
        log(f"Features shape: {X.shape}")
        log(f"Observed Label Y shape: {Y_obs.shape}   (Recidivism: 1=recid, 0=non-recid)")
        log(f"Sensitive Attribute (Race, binarized) shape: {S.shape}")
        
    elif dataset_name == "german": 
        log(f"Features shape: {X.shape}")
        log(f"Observed Label Y shape: {Y_obs.shape}   (Credit risk: 1=good, 0=bad)")
        log(f"Sensitive Attribute (Age, binarized) shape: {S.shape}")
        
    elif dataset_name == "adult":
        log(f"Features shape: {X.shape}")
        log(f"Observed Label Y shape: {Y_obs.shape}   (Label from 'income')")
        log(f"Sensitive Attribute (Sex) shape: {S.shape}")

    input_dim = X_train.shape[1]

      # One-hot encode S for adversarial model training.
    S_train_oh = tf.keras.utils.to_categorical(S_train, num_classes=2)
    S_test_oh  = tf.keras.utils.to_categorical(S_test, num_classes=2)

    ### 1. Train adversarial debiasing model (X → Y' with adversary)
    log("\nTraining adversarial model (X → Y' with adversary) ...")
    adv_model = build_adversarial_model(input_dim, lambda_adv=lambda_adv)
    # For training, we use the observed Y as target for both pseudo_Y and Y_pred.
    Y_train_obs_exp = Y_train_obs.reshape(-1, 1)
    Y_test_obs_exp  = Y_test_obs.reshape(-1, 1)
    adv_model.fit([X_train, S_train_oh],
                  {"pseudo_Y": Y_train_obs_exp, "S_pred": S_train_oh, "Y_pred": Y_train_obs_exp},  epochs=epochs, batch_size=batch_size, verbose=1)

    # Get pseudo-label predictions.
    pseudo_Y_train, S_pred, Y_pred_train = adv_model.predict([X_train, S_train_oh]) 
    pseudo_Y_test,  S_pred, Y_pred_test = adv_model.predict([X_test, S_test_oh])

    # Threshold pseudo-labels to get binary labels.
    Y_pred_train_bin = (pseudo_Y_train > 0.5).astype(np.float32)
    Y_pred_test_bin  = (pseudo_Y_test > 0.5).astype(np.float32)


    log("\nPseudo-label statistics (training):")
    for g in np.unique(S_train):
        mask = (S_train == g)
        log(f"Group {g} pseudo-positive rate: {np.mean(Y_pred_train_bin[mask]):.4f}") # average probability of a postive prediction per group -- fairness check to see if both groups receive similar treatment
 
    # Train baseline logistic regression model on observed Y (X → Y) -- regular logistic regression for baseline for comparison; does not include any fairness constraints
    log("\nTraining baseline logistic regression classifier (X → Y)...")
    baseline_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    baseline_clf.fit(X_train, Y_train_obs)
    baseline_preds = baseline_clf.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(Y_test_obs, baseline_preds)
    baseline_acc = accuracy_score(Y_test_obs, (baseline_preds > 0.5).astype(int))
    baseline_fairness = compute_fairness_metrics_manual(Y_test_obs, baseline_preds, sensitive_features=S_test)
    
    # Train fair logistic regression model on pseudo-labels (X → Y') -- using psuedo_Y from the the adv_model, 
    log("\nTraining fair logistic regression classifier (X → Y') using pseudo-labels...")
    fair_clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    fair_clf.fit(X_train, Y_pred_train_bin.ravel())
    fair_preds = fair_clf.predict_proba(X_test)[:, 1]
    fair_auc = roc_auc_score(Y_test_obs, fair_preds)
    fair_acc = accuracy_score(Y_test_obs, (fair_preds > 0.5).astype(int))
    fair_fairness = compute_fairness_metrics_manual(Y_test_obs, fair_preds, sensitive_features=S_test)

    # Aggregate metrics for plotting.
    metrics_baseline = {
        "auc": baseline_auc,
        "accuracy": baseline_acc,
        "demographic_parity_difference": baseline_fairness["demographic_parity_difference"],
        "equalized_odds_difference": baseline_fairness["equalized_odds_difference"]
    }
    metrics_fair = {
        "auc": fair_auc,
        "accuracy": fair_acc,
        "demographic_parity_difference": fair_fairness["demographic_parity_difference"],
        "equalized_odds_difference": fair_fairness["equalized_odds_difference"]
    }

    log("\nBaseline Logistic Regression (X → Y) Evaluation:")
    log(f"AUC: {baseline_auc:.4f}, Accuracy: {baseline_acc:.4f}")
    log(f"Fairness metrics: {baseline_fairness}")

    log("\nFair Logistic Regression (X → Y') Evaluation (compared to observed Y):")
    log(f"AUC: {fair_auc:.4f}, Accuracy: {fair_acc:.4f}")
    log(f"Fairness metrics: {fair_fairness}")

    # Plot comparison.
    plot_comparison(metrics_baseline, metrics_fair, plot_file_path_4, plot_file_path_2)


### UCI ADULTS

def load_and_preprocess_adult_data(data_url):
    """
    Download and preprocess the UCI Adult dataset.

    Features (X): use only:
       'age', 'education-num', 'marital-status', 'occupation', 'hours-per-week'
    Observed Label (Y): derived from 'income' (binary: 1 if '>50K', 0 otherwise)
    Sensitive attribute (S): derived from 'sex' (binary: 1 if 'Male', 0 if 'Female')

    Returns:
      X: numpy array of shape (n_samples, 5)
      Y: 1-D numpy array of observed labels.
      S: 1-D numpy array of sensitive attribute.
    """
    col_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                 "marital-status", "occupation", "relationship", "race", "sex",
                 "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    data = pd.read_csv(data_url, header=None, names=col_names, na_values=" ?", skipinitialspace=True)
    data.dropna(inplace=True)

    # Features
    feature_cols = ["age", "education-num", "marital-status", "occupation", "hours-per-week"]
    X = data[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.Categorical(X[col]).codes
    scaler = StandardScaler()
    X = scaler.fit_transform(X.values.astype(np.float32))

    # Observed label
    data['income'] = data['income'].apply(lambda s: s.replace('.', '').strip())
    y_binary = (data['income'] == '>50K').astype(np.int32)
    Y = y_binary.values  # 1-D array

    # Sensitive attribute
    S = (data['sex'].str.strip() == 'Male').astype(np.int32).values  # 1-D array

    return X, Y, S

### GERMAN 
def load_and_preprocess_german_data(data_url):
    """
    Download and preprocess the German Credit dataset.

    We assume the dataset has 21 columns.

    Features (X): Use only:
        "duration", "credit_amount", "inst_rate"
    Observed Label (Y): from "target". In many versions, target is coded as 1 for good and 2 for bad.
        We recode: good (1) -> 1, bad (2) -> 0.
    Protected Attribute (S): Use the "age" column.
        We binarize age by computing the median and setting S = 1 if age >= median (older), else 0.
    """
    col_names = ["chk_status", "duration", "credit_history", "purpose", "credit_amount",
                 "savings", "employment", "inst_rate", "personal_status_sex", "other_debtors",
                 "residence_since", "property", "age", "other_installment_plans", "housing",
                 "num_credits", "job", "num_dependents", "telephone", "foreign_worker", "target"]
    data = pd.read_csv(data_url, header=None, names=col_names, sep=' ', engine='python')

    # Features: use only duration, credit_amount, and inst_rate.
    feature_cols = ["duration", "credit_amount", "inst_rate"]
    X = data[feature_cols].copy().astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X.values)

    # Observed label: target. Recode so that 1 -> 1 (good) and 2 -> 0 (bad)
    Y = data["target"].values
    Y = np.where(Y == 1, 1, 0)

    # Protected attribute: use age.
    # Convert age to float and then binarize by the median.
    age_vals = data["age"].astype(np.float32).values
    median_age = np.median(age_vals)
    S = (age_vals >= median_age).astype(np.int32)

    return X, Y, S

### COMPAS 
def load_and_preprocess_compas_data_binary(data_url):
    """
    Download and preprocess the COMPAS dataset.

    We assume the dataset contains, among others, the following columns:
      - 'age'
      - 'race'
      - 'priors_count'
      - 'juv_fel_count'
      - 'juv_misd_count'
      - 'juv_other_count'
      - 'two_year_recid'

    Features (X): We select a few numerical features.
    Observed Label (Y): Use 'two_year_recid' as a binary label (0/1).
    Protected Attribute (S): Use 'race'. Here we binarize race so that:
         African‑American  → 1
         all other races  → 0.
    """
    data = pd.read_csv(data_url)
    # Drop rows with missing values in the selected columns.
    data = data.dropna(subset=["age", "race", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count", "two_year_recid"])

    # Observed label: two_year_recid (already 0/1)
    Y = data["two_year_recid"].values

    # Sensitive attribute: race. We set S=1 if race is African-American, else 0.
    S = (data["race"] == "African-American").astype(int).values

    # Features: use a subset of numerical features.
    feature_cols = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
    X = data[feature_cols].copy().astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X.values)

    return X, Y, S

"""
BINARY SYNTHETIC DATA EXPLORATION
"""
def generate_synthetic_data(n_samples=5000, n_features=30, bias_factor=0.4, noise_level=0.1, seed=42, multiClass=False):
    np.random.seed(seed)

    # Generate Sensitive Attribute S ~ Binomial(1, 0.5)
    S = np.random.binomial(1, 0.5, size=n_samples)

    # Generate Features X: Some function of S + Multinomial noise
    X = np.random.randn(n_samples, n_features) + 0.5 * S[:, np.newaxis]

    # Generate True Labels Y (linear function of X + noise)
    true_weights = np.random.randn(n_features)
    Y_continuous = X @ true_weights + np.random.normal(0, noise_level, size=n_samples)

    if multiClass:
        bins = np.percentile(Y_continuous, [25, 50, 75])  # 3 cut points create 4 bins
        Y = np.digitize(Y_continuous, bins=bins)  # Bins index (0,1,2,3)
        # Ensure Y is properly categorical
        Y = Y.astype(int)

    else:
        Y = np.digitize(Y_continuous, bins=np.percentile(Y_continuous, [50]))  # 2 classes (0,1) 

    X_train, X_test, Y_train_obs, Y_test_obs, S_train, S_test = train_test_split(
        X, Y, S, test_size=0.2, random_state=42
    )

    return X_train, X_test, Y_train_obs, Y_test_obs, S_train, S_test 


def inject_bias(bias_factor=0.4, seed=42, multiClass=False):
    ## will work since everything is the same seed
    np.random.seed(seed)
    if multiClass:
        X_train, X_test, Y_train_raw, Y_test_raw, S_train, S_test = generate_synthetic_data(multiClass=True)
        def apply_bias(Y, S):
            flip_mask = np.random.rand(len(Y)) < bias_factor  # Generate a flip mask for this dataset
            Y_biased = Y.copy()
    
            # Introduce bias by shifting class labels based on S
            for i in range(len(Y_biased)):
                if flip_mask[i]:
                    if S[i] == 1:  # Favoring higher classes for S=1
                        if Y_biased[i] < 3:  # Avoid exceeding class range
                            Y_biased[i] += 1  
                    elif S[i] == 0:  # Favoring lower classes for S=0
                        if Y_biased[i] > 0:  # Avoid going below class range
                            Y_biased[i] -= 1  
    
            return Y_biased

    else: 
        X_train, X_test, Y_train_raw, Y_test_raw, S_train, S_test = generate_synthetic_data()
        def apply_bias(Y, S):
            flip_mask = np.random.rand(len(Y)) < bias_factor  # Generate a flip mask for this dataset
            Y_biased = Y.copy()
            Y_biased[flip_mask & (S == 1)] = 1  # Favor positive outcomes for S=1
            Y_biased[flip_mask & (S == 0)] = 0  # Favor negative outcomes for S=0
            return Y_biased

    Y_train_biased = apply_bias(Y_train_raw, S_train)
    Y_test_biased = apply_bias(Y_test_raw, S_test)

    return Y_train_biased, Y_test_biased


def run_biased_logistic(X_train, Y_train_biased_pred, X_test, Y_test_biased_pred, Y_train_raw, Y_test_raw, S_train, S_test, multiClass=False): 
    if multiClass:
        clf = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
        clf.fit(X_train, Y_train_biased_pred)
        preds_proba = clf.predict_proba(X_test) 
        preds_class = clf.predict(X_test) # class prediction
        # multi-class AUC
        auc = roc_auc_score(to_categorical(Y_test_raw, num_classes=4), preds_proba, multi_class="ovr", average="macro")
        # multi-class accuracy
        acc = accuracy_score(Y_test_raw, preds_class)
        # fairness metrics
        fairness = multi_compute_fairness_metrics_manual(Y_test_raw, preds_proba, sensitive_features=S_test, synthetic_data=True)

    else: 
        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X_train, Y_train_biased_pred.ravel())
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test_raw, preds)
        acc = accuracy_score(Y_test_raw, (preds > 0.5).astype(int))
        fairness = compute_fairness_metrics_manual(Y_test_raw, preds, sensitive_features=S_test)
    
    return auc, acc, fairness

def run_unbiased_logistic(multiClass = False): 
    if multiClass:
        X_train, X_test, Y_train_raw, Y_test_raw, S_train, S_test = generate_synthetic_data(multiClass=True) ##  Y is multclass class

        clf = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class="multinomial")
        clf.fit(X_train, Y_train_raw)
        preds_proba = clf.predict_proba(X_test)  # multi-class probability pred
        preds_class = clf.predict(X_test)  # class pred
        # multi-class AUC
        auc = roc_auc_score(to_categorical(Y_test_raw, num_classes=4), preds_proba, multi_class="ovr", average="macro")
        # multi-class accuracy
        acc = accuracy_score(Y_test_raw, preds_class)
        # fairness metrics
        fairness = multi_compute_fairness_metrics_manual(Y_test_raw, preds_proba, sensitive_features=S_test, synthetic_data=True)

    else:
        X_train, X_test, Y_train_raw, Y_test_raw, S_train, S_test = generate_synthetic_data() ##  Y is binary class

        clf = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf.fit(X_train, Y_train_raw)
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test_raw, preds)
        acc = accuracy_score(Y_test_raw, (preds > 0.5).astype(int))
        fairness = compute_fairness_metrics_manual(Y_test_raw, preds, sensitive_features=S_test)
    
    dp_diff = fairness["demographic_parity_difference"]
    eo_diff = fairness["equalized_odds_difference"]
    
    return auc, acc, dp_diff, eo_diff


def append_text_to_image(image_path, text):
    """
    Opens an existing image and appends the given text to the bottom.
    """
    img = Image.open(image_path)
    width, height = img.size

    # Create a new image with additional height for text
    extra_height = 100  # Adjust if needed
    new_img = Image.new("RGB", (width, height + extra_height), "white")
    new_img.paste(img, (0, 0))  # Paste original image

    # Add text below the figure
    draw = ImageDraw.Draw(new_img)
    font = ImageFont.load_default()

    text_position = (10, height + 10)
    draw.text(text_position, text, fill="black", font=font)

    # Save modified image back to original path
    new_img.save(image_path)
    # log(f"Updated plot saved with appended results at: {image_path}")


# -------------------------------
# Main Function for binary synthetic data
# -------------------------------
def main_synthetic(lambda_adv=1.0, epochs=64, batch_size=128, output_dir='model_results', multiClass=False):
    set_seed(42)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # creating folder for output results
    output_path = os.path.join(repo_root, output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    log_file_path = os.path.join(output_path,'synthetic_binary_results_log.txt')
    plot_file_path_4 = os.path.join(output_path, '4_synthetic_binary_comparison_plot.png')
    plot_file_path_2 = os.path.join(output_path, '2_synthetic_binary_comparison_plot.png')

        
    def log(message):
        with open(log_file_path, 'a') as f: 
            f.write(message + '\n')

    if multiClass:
        X_train, X_test, Y_train_raw, Y_test_raw, S_train, S_test = generate_synthetic_data(multiClass=True)

    else:
        X_train, X_test, Y_train_raw, Y_test_raw, S_train, S_test = generate_synthetic_data()
        
    Y_train_biased, Y_test_biased = inject_bias(bias_factor=0.3, seed=42)

    input_dim = X_train.shape[1]

    # One-hot encode S for adversarial model training.
    S_train_oh = tf.keras.utils.to_categorical(S_train, num_classes=2)
    S_test_oh  = tf.keras.utils.to_categorical(S_test, num_classes=2)

    ### 1. Train adversarial debiasing model (X → Y' with adversary)
    log("\nTraining adversarial model (X → Y' with adversary) ...")
    adv_model = build_adversarial_model(input_dim, lambda_adv=lambda_adv)
    Y_train_biased_exp = Y_train_biased.reshape(-1, 1)
    Y_test_biased_exp  = Y_test_biased.reshape(-1, 1)
    adv_model.fit([X_train, S_train_oh],
                  {"pseudo_Y": Y_train_biased_exp, "S_pred": S_train_oh, "Y_pred": Y_train_biased_exp},
                  epochs=epochs, batch_size=batch_size, verbose=1)

    # Get predictions 
    pseudo_Y_train, S_pred, Y_pred_train = adv_model.predict([X_train, S_train_oh])
    pseudo_Y_test,  S_pred, Y_pred_test = adv_model.predict([X_test, S_test_oh])

    Y_pred_train_bin = (pseudo_Y_train > 0.5).astype(np.float32)
    Y_pred_test_bin  = (pseudo_Y_test > 0.5).astype(np.float32)


    log("\nPseudo-label statistics (training):")
    for g in np.unique(S_train):
        mask = (S_train == g)
        log(f"Group {g} pseudo-positive rate: {np.mean(Y_pred_train_bin[mask]):.4f}") 
        
    ### 2. Regular logistic regression for baseline for comparison; does not include any fairness constraints
    log("\nTraining baseline [BIASED] logistic regression classifier (X → Y)...")
    baseline_auc, baseline_acc, baseline_fairness = run_biased_logistic(X_train, Y_train_biased, X_test, Y_test_biased,  Y_train_raw, Y_test_raw, S_train, S_test)

    ### 3. Train fair logistic regression model on pseudo-labels (X → Y')
    log("\nTraining fair logistic regression classifier (X → Y') using pseudo-labels...")
    fair_auc, fair_acc, fair_fairness = run_biased_logistic(X_train, Y_pred_train_bin, X_test, Y_pred_test_bin, Y_train_raw, Y_test_raw, S_train, S_test)

    # Aggregate metrics for plotting.
    metrics_baseline = {
        "auc": baseline_auc,
        "accuracy": baseline_acc,
        "demographic_parity_difference": baseline_fairness["demographic_parity_difference"],
        "equalized_odds_difference": baseline_fairness["equalized_odds_difference"]
    }
    metrics_fair = {
        "auc": fair_auc,
        "accuracy": fair_acc,
        "demographic_parity_difference": fair_fairness["demographic_parity_difference"],
        "equalized_odds_difference": fair_fairness["equalized_odds_difference"]
    }

    log("\nBaseline Logistic Regression (X → Y) Evaluation:")
    log(f"AUC: {baseline_auc:.4f}, Accuracy: {baseline_acc:.4f}")
    log(f"Fairness metrics: {baseline_fairness}")

    log("\nFair Logistic Regression (X → Y') Evaluation (compared to observed Y):")
    log(f"AUC: {fair_auc:.4f}, Accuracy: {fair_acc:.4f}")
    log(f"Fairness metrics: {fair_fairness}")

    # Plot comparison.
    plot_comparison(metrics_baseline, metrics_fair, plot_file_path_4, plot_file_path_2)
    auc, acc, dp_diff, eo_diff = run_unbiased_logistic() # APPEND THIS TO PLOT PNG THAT IS OUTPUTTED 
    summary_text = summary_text = (
        f"Fair Unbiased Model Metrics:\n"
        f"AUC: {auc:.3f}\n"
        f"Accuracy: {acc:.3f}\n"
        f"Demographic Parity Diff: {metrics_baseline['demographic_parity_difference']:.3f}\n"
        f"Equalized Odds Diff: {metrics_baseline['equalized_odds_difference']:.3f}"
    )
    append_text_to_image(plot_file_path, summary_text)

"""
Application of Model on Multiclass for Y
"""

def multi_build_adversarial_model(input_dim, num_classes_Y, lambda_adv=1.0):
    """
    Build an adversarial debiasing model that learns pseudo‑labels Y' from X.

    Architecture:
      - Main branch (encoder): from X, several dense layers produce a latent pseudo‑label pseudo_Y (via sigmoid).
      - Adversary branch: pseudo_Y is passed through a Gradient Reversal Layer and then dense layers predict S.
      - Decoder branch: concatenates pseudo_Y and the one-hot sensitive attribute S to predict the observed label Y.

    Losses:
      - For the main branch, binary crossentropy between observed Y and pseudo_Y (and Y_pred).
      - For the adversary branch, categorical crossentropy to predict S.

    Returns a compiled Keras model that takes inputs X and S (one-hot encoded) and outputs:
      [pseudo_Y, S_pred, Y_pred].
    """
    X_input = tf.keras.Input(shape=(input_dim,), name="X")
    S_input = tf.keras.Input(shape=(2,), name="S")  # one-hot encoded S

    # Main branch: Encoder for pseudo-label.
    h = Dense(64, activation='relu')(X_input)
    h = BatchNormalization()(h)
    h = Dense(32, activation='relu')(h)
    h = BatchNormalization()(h)
    pseudo_Y = Dense(num_classes_Y, activation='softmax', name="pseudo_Y")(h) ## changed to softmax because multi-class

    # Adversary branch: from pseudo_Y, with GRL.
    """
    This is to prevent psuedo_Y from containing information about S
    - adversary will try to predict S from pseudo_Y (fair label)...if it can accurately predict S, then Y' still encodes information about S (don't want this) 
    - use the gradient reversal layer to prevent this from happening
    """
    grl = GradientReversalLayer(lambda_=lambda_adv)(pseudo_Y)
    a = Dense(32, activation='relu')(grl)
    a = BatchNormalization()(a)
    S_pred = Dense(2, activation='softmax', name="S_pred")(a)

    # Decoder branch: combine pseudo_Y and S to predict observed Y.
    concat = Concatenate()([pseudo_Y, S_input])
    d = Dense(16, activation='relu')(concat)
    d = BatchNormalization()(d)
    Y_pred = Dense(num_classes_Y, activation='softmax', name="Y_pred")(d)

    model = tf.keras.Model(inputs=[X_input, S_input],
                           outputs=[pseudo_Y, S_pred, Y_pred])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss={"pseudo_Y": "categorical_crossentropy", # changed from binary to categorical
                        "S_pred": "categorical_crossentropy",
                        "Y_pred": "categorical_crossentropy"}, # changed from binary to categorical
                  loss_weights={"pseudo_Y": 1.0, "S_pred": lambda_adv, "Y_pred": 1.0},
                  metrics={"pseudo_Y": "accuracy",
                           "S_pred": "accuracy",
                           "Y_pred": "accuracy"}) # Y_pred is the best estimate of Y accounting for fair dependencies 
    return model

def multi_compute_fairness_metrics_manual(y_true, y_pred, sensitive_features, synthetic_data = False):
    """
    Compute fairness metrics manually for multi-class classification.
    
    Args:
      y_true: Ground-truth labels (1D numpy array, categorical).
      y_pred: Predicted labels (1D numpy array, categorical).
      sensitive_features: 1D numpy array (binary sensitive attribute).
    
    Returns:
      Dictionary containing:
        - Demographic parity difference
        - Equalized odds difference
        - Selection rates per group
        - Group-wise accuracy
    """
    if synthetic_data: 
        y_pred_class = np.argmax(y_pred, axis=1)  
        groups = np.unique(sensitive_features)

        pos_rates = {g: np.zeros(4) for g in groups}  
        for g in groups:
            for c in range(4):  # Iterate through classes
                pos_rates[g][c] = np.mean(y_pred_class[sensitive_features == g] == c)
        dp_diff = np.mean(np.abs(pos_rates[0] - pos_rates[1]))  
        metrics = {g: {'tpr': np.zeros(4), 'fpr': np.zeros(4)} for g in groups}
        
        for g in groups:
            mask = (sensitive_features == g)
            y_true_g = y_true[mask]
            y_pred_g = y_pred_class[mask]
    
            for c in range(4):
                tp = np.sum((y_pred_g == c) & (y_true_g == c))
                fn = np.sum((y_pred_g != c) & (y_true_g == c))
                fp = np.sum((y_pred_g == c) & (y_true_g != c))
                tn = np.sum((y_pred_g != c) & (y_true_g != c))
                
                tpr = tp / (tp + fn + 1e-8)  # True Positive Rate for class c
                fpr = fp / (fp + tn + 1e-8)  # False Positive Rate for class c
                
                metrics[g]['tpr'][c] = tpr
                metrics[g]['fpr'][c] = fpr
    
        eo_diff = np.mean(np.abs(metrics[0]['tpr'] - metrics[1]['tpr'])) + np.mean(np.abs(metrics[0]['fpr'] - metrics[1]['fpr']))
    
        sel_rate = {g: pos_rates[g].tolist() for g in groups}
    
        group_acc = {}
        for g in groups:
            mask = (sensitive_features == g)
            group_acc[g] = accuracy_score(y_true[mask], y_pred_class[mask])
    
        return {
            "demographic_parity_difference": dp_diff,
            "equalized_odds_difference": eo_diff,
            "selection_rate": sel_rate,
            "group_accuracy": group_acc
        }
    else: 
       
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_features = np.array(sensitive_features)
    
        groups = np.unique(sensitive_features)  
        classes = np.unique(y_true)  
    
        class_rates = {g: np.zeros(len(classes)) for g in groups}
    
        for g in groups:
            mask = (sensitive_features == g)  
            for i, cl in enumerate(classes): 
                class_rates[g][i] = np.mean(y_pred[mask] == cl) 
        
        dp_diff = np.max([np.abs(class_rates[g1] - class_rates[g2]) 
                          for g1 in groups for g2 in groups if g1 != g2])
    
       
        metrics = {g: {c: {"TPR": 0, "FPR": 0} for c in classes} for g in groups}
    
        y_true = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true # categorical
    
        for g in groups:
            mask = (sensitive_features == g)
            y_true_g = y_true[mask]
            y_pred_g = y_pred[mask]
    
            for c in classes:
                tp = np.sum((y_pred_g == c) & (y_true_g == c))
                fn = np.sum((y_pred_g != c) & (y_true_g == c))
                fp = np.sum((y_pred_g == c) & (y_true_g != c))
                tn = np.sum((y_pred_g != c) & (y_true_g != c))
    
                # Avoid division by zero
                metrics[g][c]["TPR"] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics[g][c]["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
        eo_diff_vals = []
        for g1 in groups:
            for g2 in groups:
                if g1 != g2:   # trying to compare tpr and fpr across the different groupss
                    for c in classes:
                        tpr_diff = np.abs(metrics[g1][c]["TPR"] - metrics[g2][c]["TPR"])
                        fpr_diff = np.abs(metrics[g1][c]["FPR"] - metrics[g2][c]["FPR"])
                        eo_diff_vals.append(tpr_diff + fpr_diff)
    
        eo_diff = np.max(eo_diff_vals) if eo_diff_vals else 0  # Avoid empty list issue
    
        # -----------------------
        # Selection Rate Per Group
        # -----------------------
        selection_rate = {g: class_rates[g].tolist() for g in groups}
    
        # -----------------------
        # Group-Wise Accuracy
        # -----------------------
        group_acc = {}
        for g in groups:
            mask = (sensitive_features == g)
            group_acc[g] = accuracy_score(y_true[mask], y_pred[mask])
    
        return {
            "demographic_parity_difference": dp_diff,
            "equalized_odds_difference": eo_diff,
            "selection_rate": selection_rate,
            "group_accuracy": group_acc
        }

def multi_main(dataset_name="drug_multi", lambda_adv=1.0, epochs=64, batch_size=128, output_dir='model_results'):
    set_seed(42)
    
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # creating folder for output results
    output_path = os.path.join(repo_root, output_dir)
    os.makedirs(output_path, exist_ok=True)

    if dataset_name == "synthetic":
        log_file_path = os.path.join(output_path,'synthetic_multiClass_results_log.txt')
        plot_file_path_4 = os.path.join(output_path, '4_synthetic_multiClass_comparison_plot.png')
        plot_file_path_2 = os.path.join(output_path, '2_synthetic_multiClass_comparison_plot.png')

    else: 
        log_file_path = os.path.join(output_path, f'{dataset_name}_results_log.txt')
        plot_file_path_4 = os.path.join(output_path, f'4_{dataset_name}_comparison_plot.png')
        plot_file_path_2 = os.path.join(output_path, f'2_{dataset_name}_comparison_plot.png')


    open(log_file_path, 'w').close() 

    def log(message):
        with open(log_file_path, 'a') as f: 
            f.write(message + '\n')

    if dataset_name == "drug_multi":
        drug_path = "data/drug_consumption.csv" # needs to manually be added by user
        drug_path = os.path.join(repo_root, drug_path)

        X, Y_obs, S = load_and_process_drug_consumption_data(drug_path) ##  Y is multi class, S is binary
        num_classes_Y = len(np.unique(Y_obs))
        
        log(f"Loading and preprocessing {dataset_name} data...")
        X_train, X_test, Y_train_obs, Y_test_obs, S_train, S_test = train_test_split(
            X, Y_obs, S, test_size=0.2, random_state=42
        )
        
        log(f"Features shape: {X.shape}")
        log(f"Observed Label Y shape: {Y_obs.shape} (Label from 'drug consumption')")
        log(f"Sensitive Attribute (Education) shape: {S.shape}")

    elif dataset_name == "synthetic":
         # Generate multi-class synthetic dataset (Y has 4 classes, S is binary)
        X_train, X_test, Y_train_raw, Y_test_raw, S_train, S_test = generate_synthetic_data(multiClass=True)
    
        # Inject bias into Y_train and Y_test
        Y_train_biased, Y_test_biased = inject_bias(bias_factor=0.4, seed=42, multiClass=True)
        num_classes_Y = 4
        
    else: 
        return

    input_dim = X_train.shape[1]

    # One-hot encode S for adversarial model training.
    S_train_oh = tf.keras.utils.to_categorical(S_train, num_classes=2) 
    S_test_oh  = tf.keras.utils.to_categorical(S_test, num_classes=2) 

    if dataset_name == "drug_multi":

        # need to one hot encode Y
        Y_train_obs = tf.keras.utils.to_categorical(Y_train_obs, num_classes=num_classes_Y)
        Y_test_obs = tf.keras.utils.to_categorical(Y_test_obs, num_classes=num_classes_Y)
        
        Y_train_obs_1d = np.argmax(Y_train_obs, axis=1) 
        Y_test_obs_1d = np.argmax(Y_test_obs, axis=1)  
    
        ### 1. Train adversarial debiasing model (X → Y' with adversary)
        log("\nTraining adversarial model (X → Y' with adversary) ...")
        adv_model = multi_build_adversarial_model(input_dim, num_classes_Y, lambda_adv)
        adv_model.fit([X_train, S_train_oh],
                      {"pseudo_Y": Y_train_obs, "S_pred": S_train_oh, "Y_pred": Y_train_obs},
                      epochs=epochs, batch_size=batch_size, verbose=1)
        # Get pseudo-label predictions.
        pseudo_Y_train, S_pred_train, Y_pred_train = adv_model.predict([X_train, S_train_oh]) 
        pseudo_Y_test,  S_pred_test, Y_pred_test = adv_model.predict([X_test, S_test_oh])
    
        # Threshold pseudo-labels to get binary labels.
        Y_pred_train_bin = np.argmax(pseudo_Y_train, axis= 1)
        Y_pred_test_bin  = np.argmax(pseudo_Y_test, axis=1) 
    
        log("\nPseudo-label statistics (training):")
        for g in np.unique(S_train):
            mask = (S_train == g)
            log(f"Group {g} pseudo-positive rate: {np.mean(Y_pred_train_bin[mask]):.4f}")
            
        log("\nTraining baseline logistic regression classifier (X → Y)...")
        baseline_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        baseline_clf.fit(X_train, Y_train_obs_1d)
        
        baseline_preds = baseline_clf.predict_proba(X_test)    
        baseline_auc = roc_auc_score(Y_test_obs, baseline_preds, multi_class="ovr")
        baseline_preds_class = baseline_preds.argmax(axis=1)
        baseline_acc = accuracy_score(Y_test_obs_1d, baseline_preds_class)
    
        baseline_fairness = multi_compute_fairness_metrics_manual(Y_test_obs, baseline_preds_class, sensitive_features=S_test)
    
        log("\nTraining fair logistic regression classifier (X → Y') using Y_pred labels...")
        fair_clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        fair_clf.fit(X_train, Y_pred_train_bin)
        fair_preds = fair_clf.predict_proba(X_test)
        fair_auc = roc_auc_score(Y_test_obs, fair_preds, multi_class='ovr')
        fair_preds_class = fair_preds.argmax(axis=1)
        fair_acc = accuracy_score(Y_test_obs_1d, fair_preds_class)
    
        fair_fairness = multi_compute_fairness_metrics_manual(Y_test_obs, fair_preds_class, sensitive_features=S_test)

    elif dataset_name == "synthetic":
        # Convert Y to one-hot encoding for training
        Y_train_biased_oh = tf.keras.utils.to_categorical(Y_train_biased, num_classes=4)
        Y_test_biased_oh = tf.keras.utils.to_categorical(Y_test_biased, num_classes=4)

        adv_model = multi_build_adversarial_model(input_dim, num_classes_Y, lambda_adv)
        adv_model.fit([X_train, S_train_oh],
                  {"pseudo_Y": Y_train_biased_oh, "S_pred": S_train_oh, "Y_pred": Y_train_biased_oh},
                  epochs=epochs, batch_size=batch_size, verbose=1)

         # Get predictions from the adversarial model
        pseudo_Y_train, _, Y_pred_train = adv_model.predict([X_train, S_train_oh])
        pseudo_Y_test,  _, Y_pred_test = adv_model.predict([X_test, S_test_oh])
    
        # Convert softmax outputs to class predictions
        pseudo_Y_train_class = np.argmax(pseudo_Y_train, axis=1)  # Multi-class prediction
        pseudo_Y_test_class = np.argmax(pseudo_Y_test, axis=1)  # Multi-class prediction
    
        log("\nPseudo-label statistics (training):")
        for g in np.unique(S_train):
            mask = (S_train == g)
            print(f"Group {g} pseudo-class distribution: {np.bincount(pseudo_Y_train_class[mask], minlength=4)}") 
    
        log("\nTraining baseline [BIASED] logistic regression classifier (X → Y)...")
        baseline_auc, baseline_acc, baseline_fairness = run_biased_logistic(
            X_train, Y_train_biased, X_test, Y_test_biased, Y_train_raw, Y_test_raw, S_train, S_test, multiClass=True
        )
    
        log("\nTraining fair logistic regression classifier (X → Y') using pseudo-labels...")
        fair_auc, fair_acc, fair_fairness = run_biased_logistic(
            X_train, pseudo_Y_train_class, X_test, pseudo_Y_test_class, Y_train_raw, Y_test_raw, S_train, S_test, multiClass=True
        )
        auc, acc, dp_diff, eo_diff = run_unbiased_logistic(multiClass=True)
        log(f"COMPLETELY FAIR (before error injection): Baseline: AUC: {auc:.4f}, Accuracy: {acc:.4f}, Demographic Parity Difference: {dp_diff:.4f}, Equalized Odds Difference: {eo_diff:.4f}")

    # Aggregate metrics for plotting.
    metrics_baseline = {
        "auc": baseline_auc,
        "accuracy": baseline_acc,
        "demographic_parity_difference": baseline_fairness["demographic_parity_difference"],
        "equalized_odds_difference": baseline_fairness["equalized_odds_difference"]
    }
    metrics_fair = {
        "auc": fair_auc,
        "accuracy": fair_acc,
        "demographic_parity_difference": fair_fairness["demographic_parity_difference"],
        "equalized_odds_difference": fair_fairness["equalized_odds_difference"]
    }

    log("\nBaseline Logistic Regression (X → Y) Evaluation:")
    log(f"AUC: {baseline_auc:.4f}, Accuracy: {baseline_acc:.4f}")
    log(f"Fairness metrics: {baseline_fairness}")

    log("\nFair Logistic Regression (X → Y') Evaluation (compared to observed Y):")
    log(f"AUC: {fair_auc:.4f}, Accuracy: {fair_acc:.4f}")
    log(f"Fairness metrics: {fair_fairness}")

    # Plot comparison.
    plot_comparison(metrics_baseline, metrics_fair, plot_file_path_4, plot_file_path_2)

"""
Multi-Class Real World Dataset
"""
def load_and_process_drug_consumption_data(path):
    
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    
    # convert to 4 classes
    df = df[df.columns[1:]]
    df = df.replace(
        {
            "cannabis": {
                "CL0": "never_used",
                "CL1": "not_in_last_year",
                "CL2": "not_in_last_year",
                "CL3": "used_in_last_year",
                "CL4": "used_in_last_year",
                "CL5": "used_in_last_week",
                "CL6": "used_in_last_week",
            }
        }
    )
    
    educated_cat = {
        "University degree",
        "Masters degree",
        "Doctorate degree",
        "Professional certificate/ diploma"
    }
    
    df["education"] = df["education"].apply(lambda x: 1 if x in educated_cat else 0)
    
    # changing to numerical representation
    label_encoder = LabelEncoder()
    # df["age"] = label_encoder.fit_transform(df["age"]) 
    df["country"] = label_encoder.fit_transform(df["country"])
    df["ethnicity"] = label_encoder.fit_transform(df["ethnicity"])
    df["cannabis"] = label_encoder.fit_transform(df["cannabis"])

    df["gender"] = df["gender"].apply(lambda x: 1 if x == "M" else 0)
    
    X = df[df.columns[1:12]]
    Y = df["cannabis"].to_numpy()
    S = df["education"].to_numpy()
    X = X.drop(columns = ["education"])

    return X, Y, S


"""
Multi-Class: Synthetic Data
"""

"""
CROSS VALIDATION FOR REAL WORLD DATASETS
"""

def compute_fairness_metrics_cv(y_true, y_pred, sensitive_features):
    """
    Compute fairness metrics manually along with performance metrics.
    
    y_true: binary ground-truth labels (1-D numpy array).
    y_pred: continuous scores (will be thresholded at 0.5).
    sensitive_features: 1-D numpy array (0 or 1).

    Returns:
      - auc: ROC AUC score.
      - acc: overall accuracy.
      - fairness_metrics: a dictionary with:
          - Demographic parity difference (absolute difference in positive rates).
          - Equalized odds difference (average difference in TPR and FPR).
          - Selection rates per group.
          - Group-wise accuracy.
    """
    # Convert continuous predictions to binary
    y_pred_bin = (y_pred > 0.5).astype(int)

    # Compute performance metrics
    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception as e:
        auc = np.nan  # Handle cases where AUC can't be computed
    acc = accuracy_score(y_true, y_pred_bin)

    groups = np.unique(sensitive_features)

    # Demographic parity: difference in positive prediction rates between groups
    pos_rates = {}
    for g in groups: 
        pos_rates[g] = np.mean(y_pred_bin[sensitive_features == g])
    dp_diff = abs(pos_rates[0] - pos_rates[1])
    
    # Equalized odds: differences in TPR and FPR between groups
    metrics = {}
    for g in groups:
        mask = (sensitive_features == g)
        y_true_g = y_true[mask]
        y_pred_g = y_pred_bin[mask]
        tpr = np.sum((y_pred_g == 1) & (y_true_g == 1)) / (np.sum(y_true_g == 1) + 1e-8)
        fpr = np.sum((y_pred_g == 1) & (y_true_g == 0)) / (np.sum(y_true_g == 0) + 1e-8)
        metrics[g] = (tpr, fpr)
    # Average the differences in TPR and FPR between the two groups
    eo_diff = (abs(metrics[0][0] - metrics[1][0]) + abs(metrics[0][1] - metrics[1][1])) / 2.0

    # Selection rates per group (same as positive prediction rates)
    sel_rate = pos_rates

    # Group-wise accuracy
    group_acc = {}
    for g in groups:
        mask = (sensitive_features == g)
        group_acc[g] = accuracy_score(y_true[mask], y_pred_bin[mask])

    fairness_metrics = {
        "demographic_parity_difference": dp_diff,
        "equalized_odds_difference": eo_diff,
        "selection_rate": sel_rate,
        "group_accuracy": group_acc
    }
    
    return auc, acc, fairness_metrics

class AdversarialModelWrapperFixed(BaseEstimator, ClassifierMixin):
    """
    Fixed Wrapper for Adversarial Model to work with Grid Search.
    """

    def __init__(self, lambda_adv=1.0, epochs=64, batch_size=128):
        self.lambda_adv = lambda_adv
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y, S):
        """
        Train the adversarial model. S is now passed dynamically per fold.
        """
        y = y.ravel()  # Convert to 1D array
        input_dim = X.shape[1]
        S_oh = tf.keras.utils.to_categorical(S, num_classes=2)

        self.model = build_adversarial_model(input_dim, lambda_adv=self.lambda_adv)
        self.model.fit(
            [X, S_oh],
            {"pseudo_Y": y, "S_pred": S_oh, "Y_pred": y},
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self

    def predict(self, X, S):
        """
        Generate predictions from the trained model. S must match X per fold.
        """
        S_oh = tf.keras.utils.to_categorical(S, num_classes=2)
        pseudo_Y, S_pred, Y_pred = self.model.predict([X, S_oh], verbose=0)
        return (pseudo_Y > 0.5).astype(np.float32)

    def score(self, X, y, S, return_metrics=False):
        """
        Compute the optimization score combining AUC, accuracy, and fairness metrics.
        Here, predictions are generated using self.predict.
        """
        # Generate predictions on X using sensitive features S
        y_pred = self.predict(X, S)
        # Compute metrics using a (presumably) provided function
        auc, acc, fairness_metrics = compute_fairness_metrics_cv(
            y_true=y,
            y_pred=y_pred,
            sensitive_features=S
        )
        
        demographic_parity_diff = abs(fairness_metrics["demographic_parity_difference"])
        score = auc + acc - demographic_parity_diff
    
        if return_metrics:
            return score, acc, auc, demographic_parity_diff
        return score

def cross_validation_data(dataset_name, data_url, output_file):
    # Load synthetic dataset
    set_seed(42)

    if dataset_name == "compas": 
        X, Y_obs, S = load_and_preprocess_compas_data_binary(data_url)  # S is binary
    elif dataset_name == "german":
        X, Y_obs, S = load_and_preprocess_german_data(data_url)  # S is binary
    elif dataset_name == "adult":
        X, Y_obs, S = load_and_preprocess_adult_data(data_url)  # S is binary
    else:
        with open(output_file, "w") as f:
            f.write("Invalid dataset_name\n")
        return

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    output_path = os.path.join(repo_root, "cross_val_results")
    os.makedirs(output_path, exist_ok=True)
    
    output_file_path = os.path.join(output_path, output_file)

    with open(output_file_path, "w") as f:
        f.write(f"Loading and preprocessing {dataset_name} data...\n")
        
        X_train, X_test, Y_train_obs, Y_test_obs, S_train, S_test = train_test_split(
            X, Y_obs, S, test_size=0.2, random_state=42
        )
        Y_train_obs = Y_train_obs.ravel()

        param_grid = {
            "lambda_adv": [1.0, 3.0, 5.0, 7.0, 15.0],
            "epochs": [32, 64],
            "batch_size": [64, 128]
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results = []

        # Loop over all combinations of hyperparameters
        for lambda_adv, epochs, batch_size in product(param_grid["lambda_adv"],
                                                      param_grid["epochs"],
                                                      param_grid["batch_size"]):
            f.write(f"\nTesting lambda_adv={lambda_adv}, epochs={epochs}, batch_size={batch_size}\n")
            scores, accuracies, aucs, demographic_parity_diffs = [], [], [], []

            for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, Y_train_obs)):
                # Split data for the fold
                X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
                Y_train_fold, Y_val_fold = Y_train_obs[train_idx], Y_train_obs[val_idx]
                S_train_fold, S_val_fold = S_train[train_idx], S_train[val_idx]

                # Train model
                model = AdversarialModelWrapperFixed(lambda_adv=lambda_adv, epochs=epochs, batch_size=batch_size)
                model.fit(X_train_fold, Y_train_fold, S=S_train_fold)

                # Obtain predictions
                Y_train_pred = model.predict(X_train_fold, S_train_fold)
                Y_val_pred = model.predict(X_val_fold, S_val_fold)

                # Evaluate model and get metrics
                score, accuracy, auc, demographic_parity_diff = model.score(
                    X_val_fold,
                    Y_val_fold,
                    S_val_fold,
                    return_metrics=True
                )

                scores.append(score)
                accuracies.append(accuracy)
                aucs.append(auc)
                demographic_parity_diffs.append(demographic_parity_diff)

                f.write(f"  Fold {fold + 1}: Score={score:.4f}, Accuracy={accuracy:.4f}, "
                        f"AUC={auc:.4f}, Demographic Parity Diff={demographic_parity_diff:.4f}\n")

            # Average metrics across folds
            avg_score = np.mean(scores)
            avg_accuracy = np.mean(accuracies)
            avg_auc = np.mean(aucs)
            avg_demographic_parity_diff = np.mean(demographic_parity_diffs)

            results.append({
                "lambda_adv": lambda_adv,
                "epochs": epochs,
                "batch_size": batch_size,
                "score": avg_score,
                "accuracy": avg_accuracy,
                "auc": avg_auc,
                "demographic_parity_diff": avg_demographic_parity_diff
            })

            f.write(f"  Final (Avg) for lambda_adv={lambda_adv}, epochs={epochs}, batch_size={batch_size}: "
                    f"Score={avg_score:.4f}, Accuracy={avg_accuracy:.4f}, AUC={avg_auc:.4f}, "
                    f"Demographic Parity Diff={avg_demographic_parity_diff:.4f}\n")

        results_df = pd.DataFrame(results)
        best_params = results_df.loc[results_df["score"].idxmax()]
        f.write("\nBest Hyperparameters:\n")
        f.write(best_params.to_string())













