"""
Model Training Script
Trains multiple ML models for QR code scannability prediction
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class QRScannabilityPredictor:
    """
    ML Pipeline for QR Code Scannability Prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, train_path, val_path, test_path):
        """Load train, validation, and test datasets"""
        
        print("Loading datasets...")
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        print(f"✓ Train: {len(self.train_df)} samples")
        print(f"✓ Val: {len(self.val_df)} samples")
        print(f"✓ Test: {len(self.test_df)} samples")
        
    def preprocess_data(self):
        """Preprocess features and encode categorical variables"""
        
        print("\nPreprocessing data...")
        
        # Define feature columns
        numeric_features = ['strength', 'ccs_value', 'gs_value', 'prompt_length',
                           'negative_prompt_length', 'qr_version', 'image_resolution',
                           'num_iterations']
        
        categorical_features = ['error_correction_level']
        
        # Encode categorical features
        for col in categorical_features:
            le = LabelEncoder()
            self.train_df[col + '_encoded'] = le.fit_transform(self.train_df[col])
            self.val_df[col + '_encoded'] = le.transform(self.val_df[col])
            self.test_df[col + '_encoded'] = le.transform(self.test_df[col])
            self.label_encoders[col] = le
        
        # Prepare feature matrix
        feature_cols = numeric_features + [c + '_encoded' for c in categorical_features]
        
        # Add interaction features if they exist
        if 'strength_ccs_interaction' in self.train_df.columns:
            interaction_features = ['strength_ccs_interaction', 'strength_gs_interaction',
                                   'prompt_ratio']
            feature_cols.extend(interaction_features)
        
        self.feature_names = feature_cols
        
        # Extract X and y
        self.X_train = self.train_df[feature_cols].values
        self.y_train = self.train_df['is_scannable'].values
        
        self.X_val = self.val_df[feature_cols].values
        self.y_val = self.val_df['is_scannable'].values
        
        self.X_test = self.test_df[feature_cols].values
        self.y_test = self.test_df['is_scannable'].values
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"✓ Features: {len(feature_cols)}")
        print(f"✓ Feature names: {feature_cols}")
        
    def train_logistic_regression(self):
        """Train Logistic Regression (baseline)"""
        
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        self._evaluate_model(model, "Logistic Regression")
        
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        
        print("\n" + "="*60)
        print("Training Random Forest...")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        self._evaluate_model(model, "Random Forest")
        
        # Feature importance
        self._plot_feature_importance(model, "Random Forest")
        
    def train_xgboost(self):
        """Train XGBoost Classifier"""
        
        print("\n" + "="*60)
        print("Training XGBoost...")
        print("="*60)
        
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = model
        self._evaluate_model(model, "XGBoost")
        
    def train_svm(self):
        """Train Support Vector Machine"""
        
        print("\n" + "="*60)
        print("Training SVM...")
        print("="*60)
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['SVM'] = model
        self._evaluate_model(model, "SVM")
        
    def train_neural_network(self):
        """Train Multi-Layer Perceptron"""
        
        print("\n" + "="*60)
        print("Training Neural Network...")
        print("="*60)
        
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['Neural Network'] = model
        self._evaluate_model(model, "Neural Network")
        
    def _evaluate_model(self, model, model_name):
        """Evaluate model on validation and test sets"""
        
        # Validation predictions
        y_val_pred = model.predict(self.X_val)
        y_val_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Test predictions
        y_test_pred = model.predict(self.X_test)
        y_test_proba = model.predict_proba(self.X_test)[:, 1]
        
        print(f"\n{'Validation Set':^40}")
        print("-" * 40)
        print(f"Accuracy:  {accuracy_score(self.y_val, y_val_pred):.4f}")
        print(f"Precision: {precision_score(self.y_val, y_val_pred):.4f}")
        print(f"Recall:    {recall_score(self.y_val, y_val_pred):.4f}")
        print(f"F1-Score:  {f1_score(self.y_val, y_val_pred):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(self.y_val, y_val_proba):.4f}")
        
        print(f"\n{'Test Set':^40}")
        print("-" * 40)
        print(f"Accuracy:  {accuracy_score(self.y_test, y_test_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_test_pred):.4f}")
        print(f"Recall:    {recall_score(self.y_test, y_test_pred):.4f}")
        print(f"F1-Score:  {f1_score(self.y_test, y_test_pred):.4f}")
        print(f"ROC-AUC:   {roc_auc_score(self.y_test, y_test_proba):.4f}")
        
        # Confusion Matrix
        print(f"\n{'Confusion Matrix (Test)':^40}")
        print("-" * 40)
        cm = confusion_matrix(self.y_test, y_test_pred)
        print(f"TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
        print(f"FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
        
    def _plot_feature_importance(self, model, model_name):
        """Display feature importance"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\n{'Feature Importance':^40}")
            print("-" * 40)
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"{self.feature_names[idx]:30s}: {importances[idx]:.4f}")
    
    def train_all_models(self):
        """Train all models"""
        
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        self.train_svm()
        self.train_neural_network()
        
    def compare_models(self):
        """Compare all models"""
        
        print("\n" + "="*80)
        print(f"{'MODEL COMPARISON (Test Set)':^80}")
        print("="*80)
        
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1-Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_proba)
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print(results_df.to_string(index=False))
        
        # Save results
        os.makedirs('./outputs/results/metrics', exist_ok=True)
        results_df.to_csv('./outputs/results/metrics/model_comparison.csv', index=False)

        
        print(f"\n✓ Results saved to results/metrics/model_comparison.csv")
        
        return results_df
        
    def save_best_model(self):
        """
        Save the best-performing model based on ROC-AUC score
        """
        print("\nSaving best model...")

        # Ensure comparison results exist
        results_path = "./outputs/results/metrics/model_comparison.csv"
        if not os.path.exists(results_path):
            print("✗ No comparison results found.")
            return None, None

        # Load results
        results_df = pd.read_csv(results_path)

        if results_df.empty:
            print("✗ Model comparison file is empty.")
            return None, None

        # Select best model
        best_row = results_df.loc[results_df["ROC-AUC"].idxmax()]
        best_name = best_row["Model"]

        # Retrieve model object from trained models dict
        best_model = self.models.get(best_name)

        if best_model is None:
            print(f"✗ No trained model found for '{best_name}'.")
            return None, None

        # Safe output path
        model_dir = os.path.join("outputs", "models", "trained_models")
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        save_path = os.path.join(model_dir, "best_model.pkl")
        joblib.dump(best_model, save_path)

        # Also save preprocessors
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(model_dir, "label_encoders.pkl"))
        joblib.dump(self.feature_names, os.path.join(model_dir, "feature_names.pkl"))

        print(f"✓ Best model ({best_name}) saved to: {save_path}")
        return best_name, best_model



def predict_scannability(params_dict):
    """
    Make prediction for new parameters
    
    Example:
        params = {
            'strength': 0.6,
            'ccs_value': 1.5,
            'gs_value': 12.0,
            'prompt_length': 5,
            'negative_prompt_length': 10,
            'error_correction_level': 'H',
            'qr_version': 10,
            'image_resolution': 512,
            'num_iterations': 50
        }
    """
    
    # Load saved model and preprocessors
    model = joblib.load('models/trained_models/best_model.pkl')
    scaler = joblib.load('models/trained_models/scaler.pkl')
    label_encoders = joblib.load('models/trained_models/label_encoders.pkl')
    feature_names = joblib.load('models/trained_models/feature_names.pkl')
    
    # Prepare input
    input_data = []
    
    for feature in feature_names:
        if feature.endswith('_encoded'):
            col_name = feature.replace('_encoded', '')
            encoded_val = label_encoders[col_name].transform([params_dict[col_name]])[0]
            input_data.append(encoded_val)
        elif feature in params_dict:
            input_data.append(params_dict[feature])
        else:
            # Handle derived features
            if feature == 'strength_ccs_interaction':
                input_data.append(params_dict['strength'] * params_dict['ccs_value'])
            elif feature == 'strength_gs_interaction':
                input_data.append(params_dict['strength'] * params_dict['gs_value'])
            elif feature == 'prompt_ratio':
                input_data.append(params_dict['prompt_length'] / (params_dict['negative_prompt_length'] + 1))
    
    # Scale and predict
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return {
        'scannable': bool(prediction),
        'probability_not_scannable': float(probability[0]),
        'probability_scannable': float(probability[1])
    }


if __name__ == "__main__":
    # Initialize predictor
    predictor = QRScannabilityPredictor()
    
    # Load data
    predictor.load_data(
        r"C:\Users\kauzp\Downloads\TYBTECHML_Group[X]_ControlNet_QR_Education\data\processed\train.csv",
    r"C:\Users\kauzp\Downloads\TYBTECHML_Group[X]_ControlNet_QR_Education\data\processed\val.csv",
    r"C:\Users\kauzp\Downloads\TYBTECHML_Group[X]_ControlNet_QR_Education\data\processed\test.csv"

    )
    
    # Preprocess
    predictor.preprocess_data()
    
    # Train all models
    predictor.train_all_models()
    
    # Compare models
    results_df = predictor.compare_models()
    
    # Save best model
    best_name, best_model = predictor.save_best_model()
    
    # Example prediction
    print("\n" + "="*80)
    print("EXAMPLE PREDICTION")
    print("="*80)
    
    test_params = {
        'strength': 0.6,
        'ccs_value': 1.5,
        'gs_value': 12.0,
        'prompt_length': 3,
        'negative_prompt_length': 10,
        'error_correction_level': 'H',
        'qr_version': 10,
        'image_resolution': 512,
        'num_iterations': 50
    }
    
    result = predict_scannability(test_params)
    print(f"\nTest Parameters:")
    for key, value in test_params.items():
        print(f"  {key:25s}: {value}")
    
    print(f"\nPrediction:")
    print(f"  Scannable: {result['scannable']}")
    print(f"  Confidence: {result['probability_scannable']*100:.1f}%")