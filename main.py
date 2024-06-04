'''
1. main.py is directly from jupyter notebook
2. For code clarity--predict, model_training, and data_visualization is broken down into separate scripts
3. Leaving main.py if one choose to rather run in one script
'''
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data
df = pd.read_csv('D_Spacing_data.csv')

# Define features and target variables
X = df[['h', 'k', 'l', 'd - spacing']]
y = df[['Li', 'Co', 'O', '2	θ']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store models and their names
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'Neural Network': MLPRegressor(max_iter=500, random_state=42)
}

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

# Function to train and evaluate models with cross-validation and hyperparameter tuning
def train_and_evaluate(model_name, model, param_grid, X_train, y_train):
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    return best_model, cv_scores.mean(), cv_scores.std()

# Train and evaluate all models
results = {}
for name, model in models.items():
    param_grid = param_grids.get(name, None)
    trained_model, mean_cv_score, std_cv_score = train_and_evaluate(name, model, param_grid, X_train_scaled, y_train)
    print(f'{name} - Mean CV R2: {mean_cv_score:.4f}, Std CV R2: {std_cv_score:.4f}')
    results[name] = {'model': trained_model, 'mean_cv_score': mean_cv_score, 'std_cv_score': std_cv_score}

# Choose the best model based on mean CV R2 score
best_model_name = max(results, key=lambda name: results[name]['mean_cv_score'])
best_model = results[best_model_name]['model']
print(f'Best model: {best_model_name}')

# Function to predict Li, Co, O, and 2θ for a given d - spacing
def predict_values(h, k, l, d_spacing, model, scaler):
    input_data = np.array([[h, k, l, d_spacing]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Get user input and predict
h = float(input("Enter the value for h: "))
k = float(input("Enter the value for k: "))
l = float(input("Enter the value for l: "))
d_spacing = float(input("Enter the value for d - spacing: "))

predicted_values = predict_values(h, k, l, d_spacing, best_model, scaler)
print(f'Predicted values - Li: {predicted_values[0]:.4f}, Co: {predicted_values[1]:.4f}, O: {predicted_values[2]:.4f}, 2θ: {predicted_values[3]:.4f}')

# Prediction when d_spacing is less than the first row
d_spacing_less = df['d - spacing'].iloc[0] - 0.1
predicted_values_less = predict_values(h, k, l, d_spacing_less, best_model, scaler)
print(f'Predicted values for d_spacing {d_spacing_less} - Li: {predicted_values_less[0]:.4f}, Co: {predicted_values_less[1]:.4f}, O: {predicted_values_less[2]:.4f}, 2θ: {predicted_values_less[3]:.4f}')

# Visualization
sns.pairplot(df[['d - spacing', '2	θ', 'Li', 'Co', 'O']])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[['d - spacing', '2	θ', 'Li', 'Co', 'O']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()