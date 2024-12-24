import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Завантаження вашого датасету
data = pd.read_csv("diabetes_data.csv")

# X - всі ознаки, крім цільової змінної
X = data.drop(columns=["Diabetes"])  
# y - цільова змінна (діабет)
y = data["Diabetes"]

# Перетворення цільової змінної в бінарну (1 - діабет, 0 - без діабету)
y = y.astype(int)

# Створення даммі-змінних для категоріальних ознак
X = pd.get_dummies(X, drop_first=True)

# Розділення на тренувальну та тестову вибірки (70% на навчання, 30% на тестування)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, epochs=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.asarray(z, dtype=np.float64)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            
            if not isinstance(linear_model, np.ndarray):
                print(f"Error: linear_model is not a numpy array. Type: {type(linear_model)}")
                break

            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i >= self.threshold else 0 for i in y_predicted]

# Місце для збереження результатів
results = []
X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()

# Перебір гіперпараметрів для моделювання
for learning_rate in [0.01, 0.1, 1]:
    for epochs in [500, 1000, 1500]:
        for threshold in [0.4, 0.5, 0.6]:
            print(f"Start of modeling with parameters: Learning rate: {learning_rate}, Epochs: {epochs}, Threshold: {threshold};")
            model = LogisticRegressionCustom(learning_rate=learning_rate, epochs=epochs, threshold=threshold)
            model.fit(X_train_np, y_train.to_numpy())

            # Оцінка на тестових даних
            y_pred_test = model.predict(X_test_np)
            accuracy = accuracy_score(y_test, y_pred_test)

            # Збереження результатів
            results.append([learning_rate, epochs, threshold, accuracy])

# Перетворення результатів в DataFrame та збереження в CSV
results_df = pd.DataFrame(results, columns=["Learning Rate", "Epochs", "Threshold", "Accuracy"])
results_df.to_csv("custom_logistic_results_diabetes.csv", index=False)
