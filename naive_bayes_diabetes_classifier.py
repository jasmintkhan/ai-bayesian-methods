"""
naive_bayes_classifier.py

Implements a Naive Bayes classifier using categorical features: glucose and blood pressure
to predict the presence of diabetes. Includes:

- Prior and conditional probability estimation with Laplace smoothing
- Posterior probability computation
- Prediction and accuracy evaluation on test data

"""

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Naive-Bayes-Classification-Data.csv")

# Split into training and testing subsets
train_data, test_data = train_test_split(
    data, test_size=0.3, stratify=data["diabetes"], random_state=42
)

# Laplace smoothing parameter
alpha = 1

# Step 1: Compute the prior probabilities P(diabetes)
prior_probabilities = (
    train_data["diabetes"].value_counts(normalize=True).to_dict()
)
print("*** Step 1 ***\nPrior Probabilities P(diabetes):", prior_probabilities)

# Step 2: Compute the conditional probabilities P(glucose | diabetes) with Laplace smoothing
conditional_probabilities_glucose = train_data.groupby("diabetes")["glucose"].apply(
    lambda x: (x.value_counts() + alpha) / (len(x) + alpha * len(train_data["glucose"].unique()))
).unstack(fill_value=0)  # Ensure the output is always a DataFrame
print("*** Step 2 ***\nConditional Probabilities P(glucose | diabetes):")
for diabetes_value in conditional_probabilities_glucose.index:
    probabilities = conditional_probabilities_glucose.loc[diabetes_value].to_dict()
    print(f"  Diabetes = {diabetes_value}: {probabilities}")

# Step 3: Compute the conditional probabilities P(bloodpressure | diabetes) with Laplace smoothing
conditional_probabilities_bloodpressure = train_data.groupby("diabetes")["bloodpressure"].apply(
    lambda x: (x.value_counts() + alpha) / (len(x) + alpha * len(train_data["bloodpressure"].unique()))
).unstack(fill_value=0)  # Ensure the output is always a DataFrame
print("*** Step 3 ***\nConditional Probabilities P(bloodpressure | diabetes):")
for diabetes_value in conditional_probabilities_bloodpressure.index:
    probabilities = conditional_probabilities_bloodpressure.loc[diabetes_value].to_dict()
    print(f"  Diabetes = {diabetes_value}: {probabilities}")

# Function to compute P(diabetes | glucose, bloodpressure)
def compute_posterior(glucose, bloodpressure):
    posterior = {}
    for diabetes_value in prior_probabilities.keys():
        # P(glucose | diabetes)
        p_glucose_given_diabetes = conditional_probabilities_glucose.loc[
            diabetes_value, glucose
        ] if glucose in conditional_probabilities_glucose.columns else 1e-6  # Assign a small probability if unseen

        # P(bloodpressure | diabetes)
        p_bloodpressure_given_diabetes = conditional_probabilities_bloodpressure.loc[
            diabetes_value, bloodpressure
        ] if bloodpressure in conditional_probabilities_bloodpressure.columns else 1e-6  # Assign a small probability if unseen

        # P(diabetes) * P(glucose | diabetes) * P(bloodpressure | diabetes)
        posterior[diabetes_value] = (
            prior_probabilities[diabetes_value]
            * p_glucose_given_diabetes
            * p_bloodpressure_given_diabetes
        )

    # Normalize to ensure probabilities sum to 1
    total = sum(posterior.values())
    if total > 0:
        for key in posterior:
            posterior[key] /= total

    return posterior

# Step 4: Generate lookup table for P(diabetes | glucose, bloodpressure)
print("*** Step 4 ***\nLookup Table for P(diabetes | glucose, bloodpressure):")
lookup_table = {}
for glucose in train_data["glucose"].unique():
    for bloodpressure in train_data["bloodpressure"].unique():
        posterior = compute_posterior(glucose, bloodpressure)
        lookup_table[(glucose, bloodpressure)] = posterior
        print(f"  Glucose={glucose}, BloodPressure={bloodpressure} -> {posterior}")

# Step 5: Generate predictions and calculate accuracy
correct_predictions = 0
total_predictions = len(test_data)

print("*** Step 5 ***\nPredictions and Accuracy Calculation:")
for _, row in test_data.iterrows():
    glucose = row["glucose"]
    bloodpressure = row["bloodpressure"]
    actual_diabetes = row["diabetes"]

    # Compute posterior probabilities
    posterior = compute_posterior(glucose, bloodpressure)

    # Predict diabetes
    predicted_diabetes = max(posterior, key=posterior.get)

    # Print prediction results
    print(
        f"  Glucose={glucose}, BloodPressure={bloodpressure}, "
        f"Predicted={predicted_diabetes}, Actual={actual_diabetes}, Posterior={posterior}"
    )

    # Check if the prediction is correct
    if predicted_diabetes == actual_diabetes:
        correct_predictions += 1

# Calculate accuracy
accuracy = correct_predictions / total_predictions
accuracy_percentage = round(accuracy * 100, 2)
print(f"======================COMPLETE=============================\nModel Accuracy: {accuracy_percentage}%")
