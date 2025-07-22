# ai-bayesian-methods

Exploring Bayesian methods for uncertainty modeling and decision-making in AI.

This repository includes two Python implementations that apply foundational Bayesian techniques: one for inference using variable elimination in Bayesian networks, and another for classification using Naive Bayes. These projects aim to provide hands-on understanding of probabilistic reasoning in AI systems.

## Contents

### bayesian_variable_elimination.py
Implements the variable elimination algorithm on a predefined Bayesian network (alarm network). This script:
- Defines conditional probability tables (CPTs) for all variables
- Applies evidence and eliminates hidden variables to compute marginal distributions
- Outputs posterior probabilities and a formatted distribution table

  <img width="454" height="263" alt="image" src="https://github.com/user-attachments/assets/444161e6-3b74-412a-973b-a1c49899de2e" />


### naive_bayes_classifier.py
Implements a Naive Bayes classifier to predict diabetes based on categorical features (`glucose` and `bloodpressure`). The script:
- Computes prior and conditional probabilities using Laplace smoothing
- Calculates posterior probabilities for test instances
- Generates predictions and computes accuracy

### Naive-Bayes-Classification-Data.csv
The dataset used for the Naive Bayes classifier. It includes patient data with discrete glucose and blood pressure values, along with diabetes labels.

## Topics Covered

- Prior and conditional probability computation
- Laplace smoothing
- Posterior inference via Bayes’ theorem
- Variable elimination
- Factor operations: restriction, multiplication, and marginalization
- Normalization of probability distributions

## Results

### Bayesian Variable Elimination
- Computed the posterior distribution for queries such as:
  - **P(Burglary | John Calls = +j)**  
- Flexible enough to return marginal probabilities for any variable given new evidence

### Naive Bayes Classifier
- Achieved a model accuracy of **91.64%** on the test set
- Successfully printed posterior probabilities and prediction outcomes for each test instance

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-bayesian-methods.git
   cd ai-bayesian-methods
   ```

2. (Optional) Install dependencies:
   ```bash
   pip install pandas scikit-learn
   ```

3. Run each script individually:
   ```bash
   python bayesian_variable_elimination.py
   python naive_bayes_classifier.py
   ```

## License

This repository is released under the MIT License.
