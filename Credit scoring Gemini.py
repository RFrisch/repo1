# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 22:22:43 2025

@author: skarb

Gemini 2.5 Pro. Prompt: attached pdf (Consumer credit models pricing profit portfolios Thomas 2009.pdf) 
is a book about consumer credit models 
for things like credit cards, consumer loans etc. Based on this book and other information 
you gather create a complete workable scoring model in Python for scoring credit card customers 
in the Scandinavian market
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- 1. Data Simulation & Preparation ---
def create_synthetic_data(num_samples=20000):
    """Generates a synthetic DataFrame for credit scoring."""
    np.random.seed(42)
    data = {
        'age': np.random.randint(20, 70, size=num_samples),
        'income_monthly_ksek': np.random.lognormal(mean=3.5, sigma=0.5, size=num_samples).astype(int),
        'residential_status': np.random.choice(['OWNER', 'RENTER', 'LIVES_WITH_FAMILY'], p=[0.5, 0.4, 0.1], size=num_samples),
        'payment_remarks': np.random.choice([0, 1, 2, 3], p=[0.85, 0.10, 0.04, 0.01], size=num_samples),
        'existing_credits': np.random.randint(0, 10, size=num_samples),
        'credit_inquiries_last_12m': np.random.randint(0, 15, size=num_samples),
    }
    df = pd.DataFrame(data)

    prob = 1 / (1 + np.exp(-(
        -4.5
        + 0.02 * (df['age'] - 45)
        - 0.5 * np.log1p(df['income_monthly_ksek'])
        + (df['residential_status'].map({'OWNER': -0.2, 'RENTER': 0.1, 'LIVES_WITH_FAMILY': 0.3}))
        + 0.8 * df['payment_remarks']
        + 0.1 * df['existing_credits']
        + 0.15 * df['credit_inquiries_last_12m']
        + np.random.normal(0, 0.5, size=num_samples)
    )))
    df['default'] = (np.random.rand(num_samples) < prob).astype(int)
    return df

df = create_synthetic_data()
X = df.drop('default', axis=1)
y = df['default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Data simulation complete. Training set size:", len(X_train))
print("Test set size:", len(X_test))
print("Default rate in training data: {:.2%}".format(y_train.mean()))

# --- 2. Feature Engineering: Weight of Evidence (WOE) and Information Value (IV) ---
def calculate_woe_iv(df, feature, target):
    """Calculates Weight of Evidence and Information Value for a binned/categorical feature."""
    lst = []
    total_good = df[target].value_counts().get(0, 1)
    total_bad = df[target].value_counts().get(1, 1)

    for group in df[feature].unique():
        if pd.isna(group): continue
        val = df[df[feature] == group]
        n_good = len(val[val[target] == 0])
        n_bad = len(val[val[target] == 1])
        
        dist_good = n_good / total_good
        dist_bad = n_bad / total_bad
        
        if dist_good == 0 or dist_bad == 0:
            woe = np.log((dist_good + 0.0001) / (dist_bad + 0.0001))
        else:
            woe = np.log(dist_good / dist_bad)
            
        iv = (dist_good - dist_bad) * woe
        lst.append({'Value': group, 'WoE': woe, 'IV': iv})

    woe_iv_df = pd.DataFrame(lst).sort_values(by='Value')
    return woe_iv_df

woe_maps = {}
iv_values = {}
X_train_woe = pd.DataFrame(index=X_train.index)
X_test_woe = pd.DataFrame(index=X_test.index)
train_df = pd.concat([X_train, y_train], axis=1)

for feature in X_train.columns:
    if pd.api.types.is_numeric_dtype(X_train[feature]):
        binned_feature_name = feature + '_bin'
        try:
            binned_train, bin_edges = pd.qcut(train_df[feature], q=10, retbins=True, duplicates='drop')
            train_df[binned_feature_name] = binned_train
            woe_df = calculate_woe_iv(train_df, binned_feature_name, 'default')
            woe_map = dict(zip(woe_df['Value'], woe_df['WoE']))
            woe_maps[feature] = (woe_map, bin_edges)
            iv_values[feature] = woe_df['IV'].sum()
            
            X_train_woe[feature + '_woe'] = train_df.loc[X_train.index, binned_feature_name].map(woe_map)
            X_test_binned = pd.cut(X_test[feature], bins=bin_edges, include_lowest=True)
            X_test_woe[feature + '_woe'] = X_test_binned.map(woe_map)

        except ValueError:
            print(f"Skipping continuous feature {feature} due to issue with binning.")
            if feature in X_train_woe.columns: X_train_woe.drop(feature + '_woe', axis=1, inplace=True)
            if feature in X_test_woe.columns: X_test_woe.drop(feature + '_woe', axis=1, inplace=True)
            continue
    else:
        woe_df = calculate_woe_iv(train_df, feature, 'default')
        woe_map = dict(zip(woe_df['Value'], woe_df['WoE']))
        woe_maps[feature] = (woe_map, None)
        iv_values[feature] = woe_df['IV'].sum()
        
        X_train_woe[feature + '_woe'] = X_train[feature].map(woe_map)
        X_test_woe[feature + '_woe'] = X_test[feature].map(woe_map)

# CORRECTED FILLNA: Ensure all columns are numeric before filling.
# A WoE of 0 is a neutral value for unseen categories or values outside bin ranges.
X_train_woe = X_train_woe.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test_woe = X_test_woe.apply(pd.to_numeric, errors='coerce').fillna(0)


print("\nInformation Values (IV) for features:")
iv_df = pd.DataFrame(list(iv_values.items()), columns=['Feature', 'IV']).sort_values('IV', ascending=False)
print(iv_df)
print("\nFeatures transformed using Weight of Evidence.")

# --- 3. Model Training (Logistic Regression) ---
log_reg = LogisticRegression(C=1.0, solver='liblinear', random_state=42)
log_reg.fit(X_train_woe, y_train)

print("\nLogistic Regression model trained.")

# --- 4. Scorecard Generation and Scaling ---
pdo = 50
base_score = 600
factor = pdo / np.log(2)
offset = base_score - (factor * log_reg.intercept_[0])

def generate_scorecard_points(log_reg_model, woe_maps, factor, offset):
    scorecard_dict = {}
    scorecard_dict['Base Score'] = [round(offset)]
    
    for feature, (woe_map, _) in woe_maps.items():
        feature_woe_name = feature + '_woe'
        if feature_woe_name not in X_train_woe.columns: continue
        
        coef = log_reg_model.coef_[0][X_train_woe.columns.get_loc(feature_woe_name)]
        for attribute, woe in woe_map.items():
            points = -1 * (factor * coef * woe)
            scorecard_dict[f"{feature} | {attribute}"] = [round(points)]
            
    return pd.DataFrame.from_dict(scorecard_dict, orient='index', columns=['Points'])

scorecard_points = generate_scorecard_points(log_reg, woe_maps, factor, offset)
print("\n--- Generated Scorecard ---")
print(scorecard_points)

def calculate_score(data_row, scorecard_points):
    score = scorecard_points.loc['Base Score', 'Points']
    for feature in X_train.columns:
        if feature not in woe_maps: continue
        
        woe_map, bin_edges = woe_maps[feature]
        if bin_edges is not None:
            binned_value = pd.cut(pd.Series([data_row[feature]]), bins=bin_edges, include_lowest=True)[0]
            lookup_key = f"{feature} | {binned_value}"
        else:
            lookup_key = f"{feature} | {data_row[feature]}"
        
        if lookup_key in scorecard_points.index:
            score += scorecard_points.loc[lookup_key, 'Points']
            
    return score

test_scores = X_test.apply(lambda row: calculate_score(row, scorecard_points), axis=1)
print("\nSample scores for test set calculated from points table:")
print(test_scores.head())

# --- 5. Model Validation (Corrected) ---
# We use Gini and KS statistic as discussed in Chapter 2.
# The KS calculation is now corrected and integrated into the plotting function.

def plot_roc_curve_and_ks(y_true, y_pred_prob):
    """
    Calculates performance metrics and plots the ROC curve and KS statistic.
    This corrected function calculates KS directly from the ROC curve values.
    """
    # Gini Coefficient
    gini_coeff = (2 * roc_auc_score(y_true, y_pred_prob)) - 1
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    
    # KS Statistic
    ks_statistic = max(tpr - fpr)

    print(f"\nModel Performance on Test Set:")
    print(f"Gini Coefficient: {gini_coeff:.4f}")
    print(f"KS Statistic: {ks_statistic:.4f}")

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Gini = {gini_coeff:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Find the point of max KS to draw the line
    ks_max_idx = np.argmax(tpr - fpr)
    # The line is drawn from the diagonal to the curve at the point of max separation
    plt.plot([fpr[ks_max_idx], fpr[ks_max_idx]], [fpr[ks_max_idx], tpr[ks_max_idx]], 
             linestyle='--', color='red', lw=2, label=f'KS = {ks_statistic:.2f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) & KS Statistic')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, which='both', linestyle='-', linewidth=0.5)
    plt.show()

# Get predicted probabilities for the test set from the trained model
y_pred_prob = log_reg.predict_proba(X_test_woe)[:, 1]

# Call the corrected validation and plotting function
plot_roc_curve_and_ks(y_test, y_pred_prob)