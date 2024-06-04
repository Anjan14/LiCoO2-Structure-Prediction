import numpy as np
from model_training import load_data, prepare_data, train_model

# Function to predict Li, Co, O and 2θ for a given d-spacing value
def predict_values(h, k, l, d_spacing, model, scaler):
    input_data = np.array([[h, k, l, d_spacing]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

if __name__ == "__main__":
    file_path = 'D_Spacing_data.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    best_model, _ = train_model(X_train, y_train)

    # User input of the three coordinates h, l, and k
    h = float(input("Enter the value for h: ")) #print(h)
    k = float(input("Enter the value for k: ")) #print(k)
    l = float(input("Enter the value for l: ")) #print(l)
    d_spacing = float(input("Enter the value for d - spacing: ")) #print(d_spacing)

    predicted_values = predict_values(h, k, l, d_spacing, best_model, scaler)
    print(f'Predicted values - Li: {predicted_values[0]:.4f}, Co: {predicted_values[1]:.4f}, O: {predicted_values[2]:.4f}, 2θ: {predicted_values[3]:.4f}')
