import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Display first few rows to understand the data structure
print("\nFirst 5 rows of training data:")
print(train_df.head())

print("\nDataset columns:")
print(train_df.columns.tolist())

print("\nData types:")
print(train_df.dtypes)

print("\nTarget variable distribution:")
print(train_df['is_fraud'].value_counts())
print(f"Fraud percentage: {train_df['is_fraud'].mean()*100:.2f}%")

# Check for missing values
print("\nMissing values:")
print(train_df.isnull().sum())

# Basic statistics
print("\nBasic statistics:")
print(train_df.describe())

# Data Preprocessing and Feature Engineering
print("\n" + "="*50)
print("DATA PREPROCESSING AND FEATURE ENGINEERING")
print("="*50)

# Combine train and test for preprocessing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Convert trans_date_trans_time to datetime
combined_df['trans_date_trans_time'] = pd.to_datetime(combined_df['trans_date_trans_time'])

# Extract time-based features
combined_df['hour'] = combined_df['trans_date_trans_time'].dt.hour
combined_df['day_of_week'] = combined_df['trans_date_trans_time'].dt.dayofweek
combined_df['month'] = combined_df['trans_date_trans_time'].dt.month

# Convert dob to age
combined_df['dob'] = pd.to_datetime(combined_df['dob'])
combined_df['age'] = (combined_df['trans_date_trans_time'] - combined_df['dob']).dt.days // 365

# Calculate distance between merchant and cardholder (simple Euclidean distance)
combined_df['distance'] = np.sqrt((combined_df['lat'] - combined_df['merch_lat'])**2 + 
                                  (combined_df['long'] - combined_df['merch_long'])**2)

# Create categorical features
categorical_features = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    combined_df[feature + '_encoded'] = le.fit_transform(combined_df[feature].astype(str))
    label_encoders[feature] = le

# Select features for model training
feature_columns = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long', 
                   'hour', 'day_of_week', 'month', 'age', 'distance'] + \
                  [f + '_encoded' for f in categorical_features]

# Split back to train and test
train_processed = combined_df[:len(train_df)].copy()
test_processed = combined_df[len(train_df):].copy()

print(f"Selected features: {len(feature_columns)}")
print(f"Feature columns: {feature_columns}")

# Prepare training data
X_train = train_processed[feature_columns]
y_train = train_processed['is_fraud']

# Prepare test data
X_test = test_processed[feature_columns]
y_test = test_processed['is_fraud']

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Handle class imbalance using SMOTE-like approach (simple oversampling)
print("\nHandling class imbalance...")
fraud_data = train_processed[train_processed['is_fraud'] == 1]
non_fraud_data = train_processed[train_processed['is_fraud'] == 0]

print(f"Original fraud cases: {len(fraud_data)}")
print(f"Original non-fraud cases: {len(non_fraud_data)}")

# Oversample fraud cases to balance the dataset
fraud_upsampled = resample(fraud_data, 
                          replace=True,
                          n_samples=len(non_fraud_data)//5,  # Use 1:5 ratio instead of 1:1
                          random_state=42)

# Combine majority class with upsampled minority class
balanced_data = pd.concat([non_fraud_data, fraud_upsampled])
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

X_balanced = balanced_data[feature_columns]
y_balanced = balanced_data['is_fraud']

print(f"Balanced dataset shape: {X_balanced.shape}")
print(f"Balanced fraud distribution: {y_balanced.value_counts()}")

# Scale the features
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# Model Training
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Split balanced data for training and validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_balanced_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

print(f"Final training set shape: {X_train_final.shape}")
print(f"Validation set shape: {X_val.shape}")

# Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train_final)

# Train Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)
lr_model.fit(X_train_final, y_train_final)

# Model Evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Predictions on validation set
rf_val_pred = rf_model.predict(X_val)
rf_val_pred_proba = rf_model.predict_proba(X_val)[:, 1]

lr_val_pred = lr_model.predict(X_val)
lr_val_pred_proba = lr_model.predict_proba(X_val)[:, 1]

# Predictions on test set
rf_test_pred = rf_model.predict(X_test_scaled)
rf_test_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

lr_test_pred = lr_model.predict(X_test_scaled)
lr_test_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
print("RANDOM FOREST - Validation Set:")
print(classification_report(y_val, rf_val_pred))
print(f"ROC AUC Score: {roc_auc_score(y_val, rf_val_pred_proba):.4f}")

print("\nRANDOM FOREST - Test Set:")
print(classification_report(y_test, rf_test_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, rf_test_pred_proba):.4f}")

print("\nLOGISTIC REGRESSION - Validation Set:")
print(classification_report(y_val, lr_val_pred))
print(f"ROC AUC Score: {roc_auc_score(y_val, lr_val_pred_proba):.4f}")

print("\nLOGISTIC REGRESSION - Test Set:")
print(classification_report(y_test, lr_test_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test, lr_test_pred_proba):.4f}")

# Visualizations
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

plt.figure(figsize=(15, 12))

# 1. Fraud distribution
plt.subplot(2, 3, 1)
train_df['is_fraud'].value_counts().plot(kind='bar')
plt.title('Fraud Distribution in Training Data')
plt.xlabel('Is Fraud')
plt.ylabel('Count')

# 2. Transaction amount distribution
plt.subplot(2, 3, 2)
plt.hist(train_df[train_df['is_fraud']==0]['amt'], bins=50, alpha=0.7, label='Non-Fraud', density=True)
plt.hist(train_df[train_df['is_fraud']==1]['amt'], bins=50, alpha=0.7, label='Fraud', density=True)
plt.xlabel('Transaction Amount')
plt.ylabel('Density')
plt.title('Transaction Amount Distribution')
plt.legend()
plt.xlim(0, 1000)  # Focus on smaller amounts

# 3. Hour of day analysis
plt.subplot(2, 3, 3)
fraud_by_hour = train_processed.groupby(['hour', 'is_fraud']).size().unstack()
fraud_by_hour.plot(kind='bar', stacked=True)
plt.title('Fraud by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.legend(['Non-Fraud', 'Fraud'])

# 4. Confusion Matrix - Random Forest
plt.subplot(2, 3, 4)
cm_rf = confusion_matrix(y_test, rf_test_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 5. ROC Curve Comparison
plt.subplot(2, 3, 5)
# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_pred_proba)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_test_pred_proba):.3f})')

# Logistic Regression ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_test_pred_proba)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_test_pred_proba):.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()

# 6. Feature Importance (Random Forest)
plt.subplot(2, 3, 6)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importance (Random Forest)')

plt.tight_layout()
plt.show()

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Dataset loaded successfully:")
print(f"- Training samples: {len(train_df):,}")
print(f"- Test samples: {len(test_df):,}")
print(f"- Features used: {len(feature_columns)}")
print(f"- Fraud rate in training: {train_df['is_fraud'].mean()*100:.2f}%")
print(f"- Fraud rate in test: {test_df['is_fraud'].mean()*100:.2f}%")

print(f"\nBest performing model: Random Forest")
print(f"Test Set Performance:")
print(f"- ROC AUC Score: {roc_auc_score(y_test, rf_test_pred_proba):.4f}")
print(f"- Precision (Fraud): {classification_report(y_test, rf_test_pred, output_dict=True)['1']['precision']:.4f}")
print(f"- Recall (Fraud): {classification_report(y_test, rf_test_pred, output_dict=True)['1']['recall']:.4f}")
print(f"- F1-Score (Fraud): {classification_report(y_test, rf_test_pred, output_dict=True)['1']['f1-score']:.4f}")

print("\nModel training completed successfully!")
