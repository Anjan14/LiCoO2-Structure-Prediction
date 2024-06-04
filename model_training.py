import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np

# Load the csv file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to define features and target variables
def prepare_data(df):
    X = df[['h', 'k', 'l', 'd - spacing']]
    y = df[['Li', 'Co', 'O', '2	θ']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features for better readibility 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    # Dictionary to store models and thier names
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'Neural Network': MLPRegressor(max_iter=500, random_state=42)
    }

    # 2θ value is off-limits
    # Hyperparameter grids for tuning 
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt', 'log2']
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'Neural Network': {
            'hidden_layer_sizes': [(50, 50), (100, 100), (100, 50, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd']
        }
    }

    # Train and evaluate all models
    results = {}
    for name, model in models.items():
        param_grid = param_grids.get(name, None)
        
        # Train and evaluate with cross-validation and hyperparameter tuning
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
        results[name] = {'model': best_model, 'mean_cv_score': cv_scores.mean(), 'std_cv_score': cv_scores.std()}

    best_model_name = max(results, key=lambda name: results[name]['mean_cv_score'])
    best_model = results[best_model_name]['model']
    return best_model, results

if __name__ == "__main__":
    file_path = 'D_Spacing_data.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    best_model, results = train_model(X_train, y_train)
    # Print the best model
    print(f'Best model: {best_model}')
