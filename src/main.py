import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None


def prepare_data(df):
    selected_features = ['GrLivArea', 'BedroomAbvGr',
                         'YearBuilt', 'YrSold', 'SalePrice']
    df = df[selected_features].copy()

    df['Age'] = df['YrSold'] - df['YearBuilt']

    df = df[['GrLivArea', 'BedroomAbvGr', 'Age', 'SalePrice']].dropna()
    return df


def visualize_relationships(df):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='GrLivArea', y='SalePrice', data=df, alpha=0.5)
    plt.title('House Size vs Sale Price')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='Age', y='SalePrice', data=df, color='orange', alpha=0.5)
    plt.title('House Age vs Sale Price')

    plt.tight_layout()
    plt.show()


def train_linear_model(df):
    X = df[['GrLivArea', 'BedroomAbvGr', 'Age']]
    y = df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print("\n--- Model Evaluation ---")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")


def make_prediction(model, size, bedrooms, age):
    new_data = pd.DataFrame([[size, bedrooms, age]],
                            columns=['GrLivArea', 'BedroomAbvGr', 'Age'])

    prediction = model.predict(new_data)
    print(f"\n--- Prediction ---")
    print(
        f"Predicted Price for {size}sqft, {bedrooms}BR, {age}yrs old: ${prediction[0]:,.2f}")


if __name__ == "__main__":
    FILE_PATH = './datasets/train.csv'

    data = load_data(FILE_PATH)

    if data is not None:
        clean_data = prepare_data(data)
        visualize_relationships(clean_data)

        trained_model, test_X, test_y = train_linear_model(clean_data)
        evaluate_model(trained_model, test_X, test_y)

        make_prediction(trained_model, 2100, 3, 15)
