import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import random
import shap



# Load CSV file
df = pd.read_csv("Dataset_2024.csv")

# Remove spaces before column names
df.columns = df.columns.str.strip()

# Check unique values in the 'Label' column
print(df['Label'].unique())


# Plot null values as histogram
plt.figure(1, figsize=(10, 4))
plt.hist(df.isna().sum())
plt.xticks([0, 1], labels=['Not Null=0', 'Null=1'])
plt.title('Columns with Null Values')
plt.xlabel('Feature')
plt.ylabel('The number of features')
plt.show()

# Function to plot missing values
def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum()
    fig = plt.figure(figsize=(16, 5))
    missing_values.plot(kind='bar')
    plt.xlabel("Features")
    plt.ylabel("Missing values")
    plt.title("Total number of Missing values in each feature")
    plt.show()

plotMissingValues(df)


# Remove null values
data_f = df.dropna()

# Plot after removing null values
plt.figure(1, figsize=(10, 4))
plt.hist(data_f.isna().sum())
plt.title('Data after removing the Null Values')
plt.xlabel('The number of null values')
plt.ylabel('Number of columns')
plt.show()

# Replace inf and -inf with NaN
data_f = data_f.replace([np.inf, -np.inf], np.nan)

# Check for NaN and inf values
print(data_f['Label'].isnull().sum())  # Check for NaN values
print(np.isinf(data_f['Label']).sum())  # Check for inf values

# Handle NaN and inf values in 'Label' column
data_f = data_f[~data_f['Label'].isnull()]  # Remove NaN
data_f = data_f[~np.isinf(data_f['Label'])]  # Remove inf

# Plot the histogram
plt.hist(data_f['Label'], bins=2, edgecolor='black')
plt.xticks([0, 1], labels=['BENIGN=0', 'DDoS=1'])
plt.xlabel("Classes")
plt.ylabel("Count")
plt.show()

print(df.describe())

# plt.figure(figsize=(15, 10))  # Optional: Set figure size
# for col in data_f.columns:
#     plt.figure()  # Create a new figure for each histogram
#     plt.hist(data_f[col].dropna())  # Drop NaN values for plotting
#     plt.title(col)
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.show()

# Create a histogram plot for each feature
plt.figure(5)
for col in data_f.columns:
    if data_f[col].isnull().sum() == len(data_f[col]):
        print(f"Skipping {col} because it has all NaN values.")
        continue
    plt.hist(data_f[col].dropna())  # Use dropna() to avoid NaN issues
    plt.title(col)
    plt.show()
# Split data into features and target variable
X = data_f.drop('Label', axis=1)
y = data_f['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print("The train dataset size = ",X_train.shape)
print("The test dataset size = ",X_test.shape)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Getting feature importances from the trained model
importances = rf_model.feature_importances_

# Getting the indices of features sorted by importance
indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=False)
feature_names = [f"Features {i}" for i in indices]  # Replace with your column names

# Plotting feature importances horizontally
plt.figure(figsize=(8, 14))
plt.barh(range(X_train.shape[1]), importances[indices], align="center")
plt.yticks(range(X_train.shape[1]), feature_names)
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.show()
from sklearn.tree import plot_tree

estimator = rf_model.estimators_[0]  # Selecting the first estimator from the random forest model


plt.figure(figsize=(20, 10))
plot_tree(estimator, filled=True, rounded=True)
plt.show()

# Function to generate and display a detailed confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Evaluate Random Forest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

print('\nRandom Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')


# Confusion Matrix for Random Forest
plot_confusion_matrix(y_test, rf_pred, ['Benign', 'DDoS'], 'Random Forest Confusion Matrix')

# Scale the features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with increased max_iter and scaled features
lr_model = LogisticRegression(random_state=42, max_iter=200)  # Increase max_iter
lr_model.fit(X_train_scaled, y_train)  # Fit on scaled data
lr_pred = lr_model.predict(X_test_scaled)  # Predict on scaled data



lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

print('\nLogistic Regression Metrics:')
print(f'Accuracy: {lr_accuracy:.4f}')
print(f'F1 Score: {lr_f1:.4f}')
print(f'Precision: {lr_precision:.4f}')
print(f'Recall: {lr_recall:.4f}')
# Confusion Matrix for Logistic Regression
plot_confusion_matrix(y_test, lr_pred, ['Benign', 'DDoS'], 'Logistic Regression Confusion Matrix')


# Initialize the MLP (Neural Network) model
nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=200, random_state=42, verbose=True)

# Scale the features and keep column names
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Train the models
rf_model.fit(X_train_scaled, y_train)
lr_model.fit(X_train_scaled, y_train)
nn_model.fit(X_train_scaled, y_train)

# Make predictions with the trained models
rf_proba = rf_model.predict_proba(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)
nn_proba = nn_model.predict_proba(X_test_scaled)

# Make predictions with the neural network model
nn_pred = nn_model.predict(X_test_scaled)

# Now calculate accuracy, F1 score, precision, and recall using the predictions
nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)
nn_precision = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)

print('\nNeural Network Metrics:')
print(f'Accuracy: {nn_accuracy:.4f}')
print(f'F1 Score: {nn_f1:.4f}')
print(f'Precision: {nn_precision:.4f}')
print(f'Recall: {nn_recall:.4f}')
# Random Forest
rf_proba = rf_model.predict_proba(X_test)

# Logistic Regression
lr_proba = lr_model.predict_proba(X_test)



# Neural Network
nn_proba = nn_model.predict_proba(X_test)

# Combine predictions for ROC curve



# Calculate ROC curve for Random Forest
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba[:, 1])
rf_auc = auc(rf_fpr, rf_tpr)

# Calculate ROC curve for Logistic Regression
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba[:, 1])
lr_auc = auc(lr_fpr, lr_tpr)



# Calculate ROC curve for Neural Network
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_proba[:, 1])
nn_auc = auc(nn_fpr, nn_tpr)
# Plot ROC curves for all models
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')

# Plot ROC curve for random classifier (50% area)
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier (AUC = 0.50)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()

# Hybrid Model Approach
# Averaging the probabilities from Random Forest, Logistic Regression, and Neural Network
hybrid_proba = (rf_proba + lr_proba + nn_proba) / 3

# Convert averaged probabilities to class predictions (using 0.5 as threshold)
hybrid_pred = np.where(hybrid_proba[:, 1] > 0.5, 1, 0)

# Evaluate the hybrid model
hybrid_accuracy = accuracy_score(y_test, hybrid_pred)
hybrid_f1 = f1_score(y_test, hybrid_pred)
hybrid_precision = precision_score(y_test, hybrid_pred)
hybrid_recall = recall_score(y_test, hybrid_pred)

print('\nHybrid Model Metrics:')
print(f'Accuracy: {hybrid_accuracy:.4f}')
print(f'F1 Score: {hybrid_f1:.4f}')
print(f'Precision: {hybrid_precision:.4f}')
print(f'Recall: {hybrid_recall:.4f}')

# Confusion Matrix for Hybrid Model
plot_confusion_matrix(y_test, hybrid_pred, ['Benign', 'DDoS'], 'Hybrid Model Confusion Matrix')

# Calculate ROC curve for Hybrid Model
hybrid_fpr, hybrid_tpr, _ = roc_curve(y_test, hybrid_proba[:, 1])
hybrid_auc = auc(hybrid_fpr, hybrid_tpr)

# Plot ROC curve for Hybrid Model
plt.figure(figsize=(8, 6))
plt.plot(hybrid_fpr, hybrid_tpr, label=f'Hybrid Model (AUC = {hybrid_auc:.2f})')

# Plot the existing ROC curves for comparison
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier (AUC = 0.50)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Hybrid Model')
plt.legend()
plt.grid()
plt.show()

# Define the fitness function for GA
def fitness_function(individual):
    # Extract parameters from individual
    n_estimators = individual[0]
    max_depth = individual[1]

    # Create and evaluate the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Return the accuracy as the fitness score
    return accuracy_score(y_test, predictions)

# Generate initial population
def generate_population(size):
    return [[random.randint(10, 100), random.randint(1, 10)] for _ in range(size)]

# Selection process
def selection(population):
    population.sort(key=lambda x: fitness_function(x), reverse=True)
    return population[:len(population) // 2]  # Select the top half

# Crossover process
def crossover(parent1, parent2):
    child1 = [parent1[0], parent2[1]]
    child2 = [parent2[0], parent1[1]]
    return child1, child2

# Mutation process
def mutation(individual):
    if random.random() < 0.1:  # 10% chance of mutation
        individual[0] = random.randint(10, 100)  # Mutate n_estimators
    if random.random() < 0.1:  # 10% chance of mutation
        individual[1] = random.randint(1, 10)  # Mutate max_depth
    return individual

# Main GA function
def genetic_algorithm(generations=10, population_size=20):
    population = generate_population(population_size)
    for generation in range(generations):
        print(f'Generation {generation}:')
        selected = selection(population)

        # Get the best parameters and their accuracy
        best_individual = selected[0]
        best_accuracy = fitness_function(best_individual)

        print(f'  Best parameters: n_estimators = {best_individual[0]}, max_depth = {best_individual[1]}, Accuracy = {best_accuracy:.4f}')

        # Create new population
        next_population = selected[:]
        while len(next_population) < population_size:
            # Randomly select parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Perform crossover
            child1, child2 = crossover(parent1, parent2)
            
            # Perform mutation
            child1 = mutation(child1)
            child2 = mutation(child2)
            
            # Add children to the next population
            next_population.append(child1)
            next_population.append(child2)
        
        population = next_population[:population_size]  # Trim to population size

    # Return the best individual (hyperparameters) from the final population
    return selection(population)[0]

# Genetic Algorithm to find best hyperparameters for RandomForestClassifier
best_params = genetic_algorithm(generations=10, population_size=20)
best_n_estimators = best_params[0]
best_max_depth = best_params[1]

print(f'Best parameters from GA: n_estimators={best_n_estimators}, max_depth={best_max_depth}')

# Now use the best parameters to train the final Random Forest model
rf_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate the GA-tuned RandomForest
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

print('\nGA-Tuned Random Forest Metrics:')
print(f'Accuracy: {rf_accuracy:.4f}')
print(f'F1 Score: {rf_f1:.4f}')
print(f'Precision: {rf_precision:.4f}')
print(f'Recall: {rf_recall:.4f}')

from lime import lime_tabular

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print("Predictions:", predictions)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# # Initialize LIME explainer
# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=X_train.values,
#     feature_names=X_train.columns,
#     class_names=['Class 0', 'Class 1'],
#     mode='classification'
# )

# # Explain predictions for the first instance in the test set
# i = 0  # Change index for other instances if needed
# exp = explainer.explain_instance(X_test.values[i], model.predict_proba)

# # Display the explanation
# exp.show_in_notebook(show_table=True, show_all=False)
