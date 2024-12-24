import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Завантаження даних
data = pd.read_csv("diabetes_data.csv")

# Вибір ознак та цільової змінної
X = data.drop(columns=["Diabetes"])
y = data["Diabetes"]

# Перетворення категоріальних змінних в числові за допомогою one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Поділ даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабування даних
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Мінімальне та максимальне значення для нормалізації
y_min, y_max = y_train.min(), y_train.max()

# Конфігурації для шару та гіперпараметрів
layer_configs = [(64, 32), (128, 64), (64, 64, 32)]
learning_rates = [0.001, 0.01]
batch_sizes = [32, 64]
epochs = 100

# Список для збереження результатів
results = []

# Цикл по всіх конфігураціях та гіперпараметрах
for layers in layer_configs:
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Навчання моделі: Layers: {layers}, Learning Rate: {lr}, Batch Size: {batch_size}")

            # Створення моделі
            model = Sequential()
            model.add(Input(shape=(X_train.shape[1],)))  # Вхідний шар
            for layer_size in layers:
                model.add(Dense(layer_size, activation='relu'))  # Сховані шари
            model.add(Dense(1, activation='sigmoid'))  # Вихідний шар (активація sigmoid для класифікації)

            # Компільування моделі
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Навчання моделі
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

            # Прогнозування та оцінка моделі
            y_pred = model.predict(X_test).flatten()
            y_pred = (y_pred > 0.5).astype(int)  # Перетворення на 0 або 1 для класифікації
            accuracy = np.mean(y_pred == y_test)
            results.append([layers, lr, batch_size, accuracy])

# Збереження результатів у CSV файл
results_df = pd.DataFrame(results, columns=["Layer Configs", "Learning Rate", "Batch Size", "Accuracy"])
results_df.to_csv("neural_network_results_diabetes.csv", index=False)
print("Результати збережено у файл 'neural_network_results_diabetes.csv'")
