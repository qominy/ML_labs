import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("diabetes_data.csv")

# Perform one-hot encoding and drop the first column to avoid dummy variable trap
data = pd.get_dummies(data, drop_first=True)

# Separate the features and target variable
X = data.drop("Diabetes", axis=1)
y = data["Diabetes"]

# Split the dataset into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# Initialize a list to store results
results = []

# Perform hyperparameter tuning for Logistic Regression
for warm_start in [True, False]:
    for C in [0.1, 1, 10]:
        for penalty in ['l1', 'l2']:
            for max_iter in [100, 500, 1000]:
                print(f"Modeling with parameters: Warm start: {warm_start}, C: {C}, Penalty: {penalty}, Solver: liblinear, Max iter: {max_iter}")
                try:
                    # Initialize the Logistic Regression model
                    model = LogisticRegression(
                        warm_start=warm_start,
                        C=C,
                        penalty=penalty,
                        solver='liblinear',
                        max_iter=max_iter,
                        random_state=42
                    )
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Predict on the validation set
                    y_pred_val = model.predict(X_val)
                    
                    # Calculate accuracy
                    accuracy_val = accuracy_score(y_val, y_pred_val)
                    
                    # Append results
                    results.append([warm_start, C, penalty, 'liblinear', max_iter, accuracy_val])
                except Exception as e:
                    # Catch and print errors (e.g., incompatible configurations)
                    print(f"Error with parameters: Warm start: {warm_start}, C: {C}, Penalty: {penalty}, Max iter: {max_iter}. Error: {e}")

# Convert results into a DataFrame
results_df = pd.DataFrame(results, columns=['Warm_start', 'C', 'Penalty', 'Solver', 'Max_Iter', 'Accuracy'])

# Save the results to a CSV file
results_df.to_csv("model_results.csv", index=False)

print("Modeling complete. Results saved to 'model_results.csv'.")
