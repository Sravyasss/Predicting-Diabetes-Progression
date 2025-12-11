"""
Diabetes Health Indicators - IMPROVED Pipeline
Handles string labels, mixed data types, and common dataset issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class ImprovedDiabetesPipeline:
    """Improved pipeline that handles real-world data issues"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.target_encoder = None
        self.feature_encoders = {}

    def load_and_explore(self):
        """Load data and show initial exploration"""
        print("="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)

        try:
            self.df = pd.read_csv(self.data_path)
            print(f"\n✓ Data loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"\nColumns: {list(self.df.columns)}")
            print(f"\nFirst few rows:")
            print(self.df.head())

            print("\n" + "="*80)
            print("DATA EXPLORATION")
            print("="*80)

            print("\n--- Data Types ---")
            print(self.df.dtypes)

            print("\n--- Missing Values ---")
            missing = self.df.isnull().sum()
            if missing.sum() == 0:
                print("✓ No missing values")
            else:
                print(missing[missing > 0])

            print("\n--- Target Variable ---")
            if 'diabetes_stage' in self.df.columns:
                print(f"Name: diabetes_stage")
                print(f"Type: {self.df['diabetes_stage'].dtype}")
                print(f"\nValue counts:")
                print(self.df['diabetes_stage'].value_counts())
                print(f"\nUnique values: {self.df['diabetes_stage'].unique()}")

            return True

        except Exception as e:
            print(f"\n✗ Error loading data: {str(e)}")
            return False

    def clean_and_prepare(self):
        """Clean data and prepare for modeling"""
        print("\n" + "="*80)
        print("STEP 2: DATA CLEANING & PREPARATION")
        print("="*80)

        # Remove duplicates
        initial_shape = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        print(f"\n✓ Removed {initial_shape - self.df.shape[0]} duplicates")

        # Handle missing values
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)

        # Remove irrelevant columns if they exist
        cols_to_remove = ['waist_to_hip_ratio', 'screen_time_hours_per_day']
        existing_to_remove = [c for c in cols_to_remove if c in self.df.columns]
        if existing_to_remove:
            self.df = self.df.drop(columns=existing_to_remove)
            print(f"✓ Removed columns: {existing_to_remove}")

        # Identify target column
        if 'diabetes_stage' in self.df.columns:
            target_col = 'diabetes_stage'
        elif 'Diabetes_Stage' in self.df.columns:
            target_col = 'Diabetes_Stage'
        elif 'diabetes' in self.df.columns:
            target_col = 'diabetes'
        else:
            print("\n✗ Warning: Could not find target column")
            print(f"Available columns: {list(self.df.columns)}")
            return False

        # Encode target variable if it's a string
        print(f"\n--- Encoding Target Variable: {target_col} ---")
        if self.df[target_col].dtype == 'object':
            self.target_encoder = LabelEncoder()
            self.df['target_encoded'] = self.target_encoder.fit_transform(self.df[target_col])

            print(f"✓ Encoded {target_col}")
            print(f"\nClass mapping:")
            for i, class_name in enumerate(self.target_encoder.classes_):
                print(f"  {i}: {class_name}")

            # Use encoded version
            y = self.df['target_encoded']
        else:
            y = self.df[target_col]
            self.df['target_encoded'] = y

        # Separate features
        feature_cols = [c for c in self.df.columns if c not in [target_col, 'target_encoded']]
        X = self.df[feature_cols].copy()

        print(f"\n--- Processing Features ---")
        print(f"Total features: {len(feature_cols)}")

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        print(f"Categorical: {len(categorical_cols)}")
        print(f"Numerical: {len(numerical_cols)}")

        # Encode categorical features
        if categorical_cols:
            print(f"\n✓ Encoding categorical features...")
            for col in categorical_cols:
                if X[col].nunique() < 20:  # One-hot encode if few categories
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
                else:  # Label encode if many categories
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.feature_encoders[col] = le

        # Store prepared data
        self.X = X
        self.y = y

        print(f"\n✓ Final feature count: {X.shape[1]}")
        print(f"✓ Final sample count: {X.shape[0]}")

        return True

    def feature_analysis(self):
        """Analyze feature importance via correlation"""
        print("\n" + "="*80)
        print("STEP 3: FEATURE ANALYSIS")
        print("="*80)

        # Create temporary dataframe with all numerical features
        temp_df = self.X.copy()
        temp_df['target'] = self.y

        # Calculate correlations
        print("\n--- Feature Correlations with Target ---")
        correlations = temp_df.corr()['target'].abs().sort_values(ascending=False)
        print(correlations.head(15))

        # Remove very low correlation features
        low_corr = correlations[correlations < 0.01].index.tolist()
        low_corr = [f for f in low_corr if f != 'target']

        if low_corr:
            print(f"\n✓ Removing {len(low_corr)} features with correlation < 0.01")
            self.X = self.X.drop(columns=low_corr)
            print(f"Remaining features: {self.X.shape[1]}")

    def split_and_scale(self):
        """Split data and scale features"""
        print("\n" + "="*80)
        print("STEP 4: TRAIN-TEST SPLIT & SCALING")
        print("="*80)

        # Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"\n✓ Split completed")
        print(f"Training: {self.X_train.shape[0]} samples")
        print(f"Testing: {self.X_test.shape[0]} samples")
        print(f"Features: {self.X_train.shape[1]}")

        print("\n--- Class Distribution (Training) ---")
        print(pd.Series(self.y_train).value_counts().sort_index())

        # Scale
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )

        print("\n✓ Features scaled")

    def balance_classes(self):
        """Apply SMOTE for class imbalance"""
        print("\n" + "="*80)
        print("STEP 5: HANDLING CLASS IMBALANCE")
        print("="*80)

        print("\n--- Before SMOTE ---")
        print(pd.Series(self.y_train).value_counts().sort_index())

        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, pd.Series(self.y_train).value_counts().min() - 1))
            self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
                self.X_train_scaled, self.y_train
            )

            print("\n--- After SMOTE ---")
            print(pd.Series(self.y_train_balanced).value_counts().sort_index())
            print(f"\n✓ Balanced dataset size: {len(self.y_train_balanced)}")

        except Exception as e:
            print(f"\n⚠ SMOTE failed: {str(e)}")
            print("Using original (imbalanced) data")
            self.X_train_balanced = self.X_train_scaled
            self.y_train_balanced = self.y_train

    def train_models(self):
        """Train multiple models"""
        print("\n" + "="*80)
        print("STEP 6: MODEL TRAINING")
        print("="*80)

        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=10, random_state=42, eval_metric='mlogloss'),
            'CatBoost': CatBoostClassifier(iterations=100, depth=10, random_state=42, verbose=False)
        }

        print(f"\nTraining {len(self.models)} models...")

        for name, model in self.models.items():
            print(f"\n{name}...", end=" ")
            try:
                model.fit(self.X_train_balanced, self.y_train_balanced)
                print("✓")
            except Exception as e:
                print(f"✗ Error: {str(e)}")

    def evaluate_models(self):
        """Evaluate all models"""
        print("\n" + "="*80)
        print("STEP 7: MODEL EVALUATION")
        print("="*80)

        results = []

        for name, model in self.models.items():
            try:
                print(f"\n--- {name} ---")

                y_pred = model.predict(self.X_test_scaled)

                acc = accuracy_score(self.y_test, y_pred)
                f1_macro = f1_score(self.y_test, y_pred, average='macro')
                f1_weighted = f1_score(self.y_test, y_pred, average='weighted')

                print(f"Accuracy: {acc:.4f}")
                print(f"F1-Score (Macro): {f1_macro:.4f}")
                print(f"F1-Score (Weighted): {f1_weighted:.4f}")

                results.append({
                    'Model': name,
                    'Accuracy': acc,
                    'F1_Macro': f1_macro,
                    'F1_Weighted': f1_weighted
                })

                # Show classification report for best model
                if len(results) <= 2:  # Show for first few models
                    print("\nClassification Report:")

                    # Get class names if we have encoder
                    if self.target_encoder:
                        target_names = self.target_encoder.classes_
                    else:
                        target_names = [str(i) for i in sorted(self.y.unique())]

                    print(classification_report(self.y_test, y_pred, target_names=target_names))

            except Exception as e:
                print(f"✗ Error evaluating {name}: {str(e)}")

        # Results summary
        self.results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(self.results_df.to_string(index=False))

        # Best model
        best = self.results_df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best['Model']}")
        print(f"   Accuracy: {best['Accuracy']:.4f}")
        print(f"   F1-Score: {best['F1_Macro']:.4f}")

        return self.results_df

    def save_results(self, output_dir='output'):
        """Save results"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        # Save metrics
        results_path = f"{output_dir}/model_results.csv"
        self.results_df.to_csv(results_path, index=False)
        print(f"✓ Saved: {results_path}")

        # Save best model
        import joblib
        best_model_name = self.results_df.iloc[0]['Model']
        best_model = self.models[best_model_name]

        model_path = f"{output_dir}/best_model.pkl"
        scaler_path = f"{output_dir}/scaler.pkl"

        joblib.dump(best_model, model_path)
        joblib.dump(self.scaler, scaler_path)

        print(f"✓ Saved: {model_path}")
        print(f"✓ Saved: {scaler_path}")

        # Save feature list
        feature_path = f"{output_dir}/features.txt"
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.X.columns))
        print(f"✓ Saved: {feature_path}")

        # Save confusion matrix for best model
        best_model = self.models[best_model_name]
        y_pred = best_model.predict(self.X_test_scaled)
        cm = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(10, 8))

        # Get labels
        if self.target_encoder:
            labels = self.target_encoder.classes_
        else:
            labels = [str(i) for i in sorted(self.y.unique())]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        cm_path = f"{output_dir}/confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {cm_path}")

        print(f"\n✓ All results saved to: {output_dir}/")

    def run_pipeline(self):
        """Run complete pipeline"""
        print("\n" + "="*80)
        print("DIABETES PREDICTION - COMPLETE PIPELINE")
        print("="*80)

        if not self.load_and_explore():
            return False

        if not self.clean_and_prepare():
            return False

        self.feature_analysis()
        self.split_and_scale()
        self.balance_classes()
        self.train_models()
        self.evaluate_models()
        self.save_results()

        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)

        return True


if __name__ == "__main__":
    # Run pipeline
    data_file = "diabetes_dataset.csv"

    pipeline = ImprovedDiabetesPipeline(data_file)
    pipeline.run_pipeline()