import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and read the data
def load_data(file_path):
    return pd.read_csv(file_path)

# Visualition with pyplot
def plot_pairplot(df):
    sns.pairplot(df[['d - spacing', '2	θ', 'Li', 'Co', 'O']])
    plt.show()

# Visualization with correlation heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['d - spacing', '2	θ', 'Li', 'Co', 'O']].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

# Add other visualizations later?

if __name__ == "__main__":
    file_path = 'D_Spacing_data.csv'
    df        = load_data(file_path)
    plot_pairplot(df)
    plot_correlation_heatmap(df)