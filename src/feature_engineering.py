"""
Feature Engineering Module
Creates derived features and transformations
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for QR code scannability prediction
    """
    
    def __init__(self):
        self.poly = None
        
    def add_interaction_features(self, df):
        """
        Add interaction features between important parameters
        
        Based on paper findings:
        - Strength × CCS: Combined effect on generation
        - Strength × GS: Guidance interaction
        - Prompt ratio: Balance of positive/negative prompts
        """
        print("\nAdding interaction features...")
        
        df = df.copy()
        
        # Multiplicative interactions
        if 'strength' in df.columns and 'ccs_value' in df.columns:
            df['strength_ccs_interaction'] = df['strength'] * df['ccs_value']
            print("✓ Added strength_ccs_interaction")
        
        if 'strength' in df.columns and 'gs_value' in df.columns:
            df['strength_gs_interaction'] = df['strength'] * df['gs_value']
            print("✓ Added strength_gs_interaction")
        
        if 'ccs_value' in df.columns and 'gs_value' in df.columns:
            df['ccs_gs_interaction'] = df['ccs_value'] * df['gs_value']
            print("✓ Added ccs_gs_interaction")
        
        # Ratio features
        if 'prompt_length' in df.columns and 'negative_prompt_length' in df.columns:
            df['prompt_ratio'] = df['prompt_length'] / (df['negative_prompt_length'] + 1)
            print("✓ Added prompt_ratio")
        
        return df
    
    def add_polynomial_features(self, df, degree=2, features=None):
        """
        Add polynomial features (squares, cubes, etc.)
        
        Parameters:
        -----------
        degree : int
            Polynomial degree (2 = squares, 3 = cubes)
        features : list
            Features to polynomialize (None = key features)
        """
        print(f"\nAdding polynomial features (degree={degree})...")
        
        df = df.copy()
        
        if features is None:
            features = ['strength', 'ccs_value', 'gs_value']
        
        for feature in features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    new_col = f'{feature}_pow{d}'
                    df[new_col] = df[feature] ** d
                    print(f"✓ Added {new_col}")
        
        return df
    
    def add_binned_features(self, df):
        """
        Create categorical bins for continuous features
        Based on paper's optimal ranges
        """
        print("\nAdding binned features...")
        
        df = df.copy()
        
        # Strength categories
        if 'strength' in df.columns:
            df['strength_category'] = pd.cut(
                df['strength'],
                bins=[0, 0.5, 0.7, 1.0],
                labels=['low', 'optimal', 'high'],
                include_lowest=True
            )
            print("✓ Added strength_category")
        
        # CCS categories
        if 'ccs_value' in df.columns:
            df['ccs_category'] = pd.cut(
                df['ccs_value'],
                bins=[0, 1.3, 1.7, 3.0],
                labels=['low', 'optimal', 'high'],
                include_lowest=True
            )
            print("✓ Added ccs_category")
        
        # Prompt length categories
        if 'prompt_length' in df.columns:
            df['prompt_category'] = pd.cut(
                df['prompt_length'],
                bins=[0, 3, 10, 100],
                labels=['short', 'medium', 'long'],
                include_lowest=True
            )
            print("✓ Added prompt_category")
        
        # GS categories
        if 'gs_value' in df.columns:
            df['gs_category'] = pd.cut(
                df['gs_value'],
                bins=[0, 10, 15, 30],
                labels=['low', 'optimal', 'high'],
                include_lowest=True
            )
            print("✓ Added gs_category")
        
        # Resolution categories
        if 'image_resolution' in df.columns:
            df['resolution_category'] = pd.cut(
                df['image_resolution'],
                bins=[0, 512, 768, 2000],
                labels=['low', 'medium', 'high'],
                include_lowest=True
            )
            print("✓ Added resolution_category")
        
        return df
    
    def add_domain_features(self, df):
        """
        Add domain-specific features based on paper insights
        """
        print("\nAdding domain-specific features...")
        
        df = df.copy()
        
        # Optimal parameter indicator (from paper)
        if all(col in df.columns for col in ['strength', 'ccs_value', 'prompt_length']):
            df['is_optimal_params'] = (
                (df['strength'] >= 0.5) & (df['strength'] <= 0.7) &
                (df['ccs_value'] >= 1.3) & (df['ccs_value'] <= 1.7) &
                (df['prompt_length'] <= 3)
            ).astype(int)
            print("✓ Added is_optimal_params")
        
        # Suboptimal parameter count
        if all(col in df.columns for col in ['strength', 'ccs_value', 'gs_value', 'prompt_length']):
            suboptimal_count = 0
            suboptimal_count += ((df['strength'] < 0.5) | (df['strength'] > 0.7)).astype(int)
            suboptimal_count += ((df['ccs_value'] < 1.3) | (df['ccs_value'] > 1.7)).astype(int)
            suboptimal_count += ((df['gs_value'] < 10) | (df['gs_value'] > 15)).astype(int)
            suboptimal_count += (df['prompt_length'] > 10).astype(int)
            df['suboptimal_count'] = suboptimal_count
            print("✓ Added suboptimal_count")
        
        # Parameter deviation from optimal
        if 'strength' in df.columns:
            optimal_strength = 0.6
            df['strength_deviation'] = np.abs(df['strength'] - optimal_strength)
            print("✓ Added strength_deviation")
        
        if 'ccs_value' in df.columns:
            optimal_ccs = 1.5
            df['ccs_deviation'] = np.abs(df['ccs_value'] - optimal_ccs)
            print("✓ Added ccs_deviation")
        
        if 'gs_value' in df.columns:
            optimal_gs = 12.5
            df['gs_deviation'] = np.abs(df['gs_value'] - optimal_gs)
            print("✓ Added gs_deviation")
        
        # Prompt complexity score
        if all(col in df.columns for col in ['prompt_length', 'negative_prompt_length']):
            df['prompt_complexity'] = (
                df['prompt_length'] * 0.7 + df['negative_prompt_length'] * 0.3
            )
            print("✓ Added prompt_complexity")
        
        return df
    
    def add_statistical_features(self, df):
        """
        Add statistical aggregations across parameters
        """
        print("\nAdding statistical features...")
        
        df = df.copy()
        
        # Select numeric columns for aggregation
        numeric_cols = ['strength', 'ccs_value', 'gs_value']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) >= 2:
            # Mean of key parameters
            df['params_mean'] = df[available_cols].mean(axis=1)
            print("✓ Added params_mean")
            
            # Standard deviation of key parameters
            df['params_std'] = df[available_cols].std(axis=1)
            print("✓ Added params_std")
            
            # Range of key parameters
            df['params_range'] = df[available_cols].max(axis=1) - df[available_cols].min(axis=1)
            print("✓ Added params_range")
        
        return df
    
    def add_logarithmic_features(self, df, features=None):
        """
        Add log transformations for skewed features
        """
        print("\nAdding logarithmic features...")
        
        df = df.copy()
        
        if features is None:
            features = ['prompt_length', 'negative_prompt_length', 'num_iterations']
        
        for feature in features:
            if feature in df.columns:
                # Add 1 to avoid log(0)
                new_col = f'{feature}_log'
                df[new_col] = np.log1p(df[feature])
                print(f"✓ Added {new_col}")
        
        return df
    
    def add_all_features(self, df):
        """
        Apply all feature engineering steps
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        initial_features = len(df.columns)
        
        # Apply all transformations
        df = self.add_interaction_features(df)
        df = self.add_binned_features(df)
        df = self.add_domain_features(df)
        df = self.add_statistical_features(df)
        df = self.add_logarithmic_features(df)
        
        final_features = len(df.columns)
        
        print("\n" + "="*60)
        print(f"Feature engineering complete!")
        print(f"Features: {initial_features} → {final_features} (+{final_features - initial_features})")
        print("="*60)
        
        return df
    
    def select_features(self, df, method='correlation', threshold=0.1, target_col='is_scannable'):
        """
        Feature selection using correlation or other methods
        
        Parameters:
        -----------
        method : str
            'correlation', 'variance', or 'manual'
        threshold : float
            Minimum correlation with target
        """
        print(f"\nSelecting features (method={method}, threshold={threshold})...")
        
        if method == 'correlation' and target_col in df.columns:
            # Calculate correlation with target
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != target_col]
            
            correlations = df[numeric_cols].corrwith(df[target_col]).abs()
            selected = correlations[correlations >= threshold].index.tolist()
            
            print(f"✓ Selected {len(selected)} features with |correlation| >= {threshold}")
            print(f"  Top 5 correlated: {correlations.nlargest(5).to_dict()}")
            
            return selected
        
        elif method == 'variance':
            from sklearn.feature_selection import VarianceThreshold
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != target_col]
            
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(df[numeric_cols])
            
            selected = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
            
            print(f"✓ Selected {len(selected)} features with variance >= {threshold}")
            
            return selected
        
        return None


def engineer_features_pipeline(train_df, val_df, test_df):
    """
    Complete feature engineering pipeline for train, val, test
    
    Returns:
    --------
    train_df, val_df, test_df (with engineered features)
    """
    engineer = FeatureEngineer()
    
    print("\n" + "="*60)
    print("ENGINEERING FEATURES FOR ALL SPLITS")
    print("="*60)
    
    # Apply to all splits
    print("\n>>> Training Set")
    train_df = engineer.add_all_features(train_df)
    
    print("\n>>> Validation Set")
    val_df = engineer.add_all_features(val_df)
    
    print("\n>>> Test Set")
    test_df = engineer.add_all_features(test_df)
    
    return train_df, val_df, test_df, engineer


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Engineer features
    train_df, val_df, test_df, engineer = engineer_features_pipeline(
        train_df, val_df, test_df
    )
    
    # Save engineered data
    print("\nSaving engineered datasets...")
    train_df.to_csv('data/processed/train_engineered.csv', index=False)
    val_df.to_csv('data/processed/val_engineered.csv', index=False)
    test_df.to_csv('data/processed/test_engineered.csv', index=False)
    
    print("\n✓ Feature engineering complete!")
    print(f"Final feature count: {len(train_df.columns)}")
