"""
Data Preprocessing Module
Handles data cleaning, encoding, and transformation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, filepath):
        """Load dataset from CSV"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} samples with {len(df.columns)} columns")
        return df
    
    def inspect_data(self, df):
        """Display data summary"""
        print("\n" + "="*60)
        print("DATA INSPECTION")
        print("="*60)
        
        print(f"\nShape: {df.shape}")
        print(f"\nColumns: {list(df.columns)}")
        
        print("\nData Types:")
        print(df.dtypes)
        
        print("\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values")
        
        print("\nBasic Statistics:")
        print(df.describe())
        
        if 'is_scannable' in df.columns:
            print("\nTarget Distribution:")
            print(df['is_scannable'].value_counts())
            print(f"Scannable: {df['is_scannable'].mean()*100:.1f}%")
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Handle missing values
        
        Parameters:
        -----------
        strategy : str
            'mean', 'median', 'mode', or 'drop'
        """
        print(f"\nHandling missing values (strategy: {strategy})...")
        
        initial_rows = len(df)
        missing_before = df.isnull().sum().sum()
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mode':
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        missing_after = df.isnull().sum().sum()
        
        print(f"✓ Missing values: {missing_before} → {missing_after}")
        print(f"✓ Rows: {initial_rows} → {len(df)}")
        
        return df
    
    def remove_outliers(self, df, columns=None, method='iqr', threshold=1.5):
        """
        Remove outliers using IQR or Z-score method
        
        Parameters:
        -----------
        columns : list
            Columns to check for outliers (None = all numeric)
        method : str
            'iqr' or 'zscore'
        threshold : float
            IQR multiplier (default 1.5) or Z-score threshold (default 3)
        """
        print(f"\nRemoving outliers (method: {method})...")
        
        initial_rows = len(df)
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
            columns = [c for c in columns if c != 'is_scannable']
        
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                z_scores = np.abs(stats.zscore(df[col]))
                df = df[z_scores < threshold]
        
        print(f"✓ Rows: {initial_rows} → {len(df)} ({initial_rows - len(df)} outliers removed)")
        
        return df
    
    def encode_categorical(self, df, categorical_cols=None):
        """
        Encode categorical variables
        
        Parameters:
        -----------
        categorical_cols : list
            List of categorical column names
        """
        print("\nEncoding categorical variables...")
        
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns
            categorical_cols = [c for c in categorical_cols if c not in ['generated_date', 'sample_id']]
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
                print(f"✓ Encoded {col}: {len(le.classes_)} unique values")
        
        return df
    
    def scale_features(self, X_train, X_val=None, X_test=None, method='standard'):
        """
        Scale features
        
        Parameters:
        -----------
        method : str
            'standard' (z-score) or 'minmax' (0-1 range)
        """
        print(f"\nScaling features (method: {method})...")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'standard' or 'minmax'")
        
        X_train_scaled = scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            results.append(X_test_scaled)
        
        self.scaler = scaler
        print("✓ Features scaled successfully")
        
        return results if len(results) > 1 else results[0]
    
    def prepare_features(self, df, target_col='is_scannable', exclude_cols=None):
        """
        Prepare feature matrix and target vector
        
        Parameters:
        -----------
        df : DataFrame
            Input dataframe
        target_col : str
            Target column name
        exclude_cols : list
            Columns to exclude from features
        """
        print("\nPreparing features...")
        
        if exclude_cols is None:
            exclude_cols = ['generated_date', 'sample_id', 'prompt_category', 
                          'strength_category', 'resolution_category']
        
        # Remove original categorical columns (keep encoded versions)
        categorical_originals = [col for col in df.columns 
                                if col + '_encoded' in df.columns]
        exclude_cols.extend(categorical_originals)
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + [target_col]]
        
        # Extract features and target
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else None
        
        self.feature_names = feature_cols
        
        print(f"✓ Features: {len(feature_cols)}")
        print(f"  Columns: {feature_cols}")
        
        if y is not None:
            print(f"✓ Target: {target_col}")
            print(f"  Distribution: {np.bincount(y.astype(int))}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into train, validation, and test sets
        
        Parameters:
        -----------
        test_size : float
            Proportion for test set
        val_size : float
            Proportion for validation set (from remaining after test)
        """
        print(f"\nSplitting data (test={test_size}, val={val_size})...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"✓ Train: {len(X_train)} samples")
        print(f"✓ Val: {len(X_val)} samples")
        print(f"✓ Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_preprocessing_pipeline(self):
        """Return fitted preprocessing objects"""
        return {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }


def preprocess_full_pipeline(train_path, val_path, test_path):
    """
    Complete preprocessing pipeline
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    """
    preprocessor = DataPreprocessor()
    
    # Load data
    train_df = preprocessor.load_data(train_path)
    val_df = preprocessor.load_data(val_path)
    test_df = preprocessor.load_data(test_path)
    
    # Inspect
    preprocessor.inspect_data(train_df)
    
    # Handle missing values
    train_df = preprocessor.handle_missing_values(train_df)
    val_df = preprocessor.handle_missing_values(val_df)
    test_df = preprocessor.handle_missing_values(test_df)
    
    # Encode categorical
    train_df = preprocessor.encode_categorical(train_df, ['error_correction_level'])
    val_df = preprocessor.encode_categorical(val_df, ['error_correction_level'])
    test_df = preprocessor.encode_categorical(test_df, ['error_correction_level'])
    
    # Prepare features
    X_train, y_train = preprocessor.prepare_features(train_df)
    X_val, y_val = preprocessor.prepare_features(val_df)
    X_test, y_test = preprocessor.prepare_features(test_df)
    
    # Scale features
    X_train, X_val, X_test = preprocessor.scale_features(X_train, X_val, X_test)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    # Example usage
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_full_pipeline(
        r"C:\Users\kauzp\Downloads\TYBTECHML_Group[X]_ControlNet_QR_Education\data\processed\train.csv",
        r"C:\Users\kauzp\Downloads\TYBTECHML_Group[X]_ControlNet_QR_Education\data\processed\val.csv",
        r"C:\Users\kauzp\Downloads\TYBTECHML_Group[X]_ControlNet_QR_Education\data\processed\test.csv"
    )
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"X_test: {X_test.shape}")
