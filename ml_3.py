import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from catboost import CatBoostRegressor

# Завантаження вашого датасету
data = pd.read_csv("diabetes_data.csv")

# Вибір ознак і цільової змінної
X = data.drop(columns=["Diabetes"])  # Замість "Survival_Months" беремо колонку "Diabetes"
y = data["Diabetes"]  # Цільова змінна

# Оскільки в вашому датасеті можуть бути категоріальні змінні, застосуємо one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Розділення даних на тренувальні і тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Для нормалізації метрики використаємо мінімум і максимум з тренувального набору
y_min, y_max = y_train.min(), y_train.max()

# Гіперпараметри для пошуку оптимальної моделі
learning_rates = [0.01, 0.05, 0.1]
depths = [4, 6, 8]
iterations = [500, 1000, 1500]
results = []

# Пошук найкращих параметрів
for lr in learning_rates:
    for depth in depths:
        for iter_count in iterations:
            print(f"Моделювання з параметрами: Learning rate: {lr}, Depth: {depth}, Iterations: {iter_count}")
            model = CatBoostRegressor(
                learning_rate=lr,
                depth=depth,
                iterations=iter_count,
                verbose=0,
                random_seed=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Обчислення RMSE і NRMSE
            rmse = root_mean_squared_error(y_test, y_pred)
            nrmse = rmse / (y_max - y_min)
            results.append([lr, depth, iter_count, rmse, nrmse])

# Створення DataFrame з результатами
results_df = pd.DataFrame(results, columns=["Learning Rate", "Depth", "Iterations", "RMSE", "NRMSE"])

# Запис результатів у CSV
results_df.to_csv("catboost_regression_results.csv", index=False)
