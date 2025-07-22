"""
bayesian_variable_elimination.py

Implements variable elimination on a simple Bayesian Network to infer
P(Burglary | JohnCalls = +j). Factors are manually defined using CPTs,
and the script performs:
- Applying evidence
- Multiplying and marginalizing factors
- Normalizing final probabilities

Network Structure:
    B: Burglary
    E: Earthquake
    A: Alarm
    J: John Calls
    M: Mary Calls
"""

# Conditional Probability Tables as dictionaries
CPT_B = {  # Variables: ["B"]
    ("+b",): 0.001, 
    ("-b",): 0.999
}
CPT_E = {  # Variables: ["E"]
    ("+e",): 0.002, 
    ("-e",): 0.998
}
CPT_A = {  # Variables: ["B", "E", "A"]
    ("+b", "+e", "+a"): 0.95,
    ("+b", "+e", "-a"): 0.05,
    ("+b", "-e", "+a"): 0.94,
    ("+b", "-e", "-a"): 0.06,
    ("-b", "+e", "+a"): 0.29,
    ("-b", "+e", "-a"): 0.71,
    ("-b", "-e", "+a"): 0.001,
    ("-b", "-e", "-a"): 0.999,
}
CPT_J = {  # Variables: ["A", "J"]
    ("+a", "+j"): 0.9,
    ("+a", "-j"): 0.1,
    ("-a", "+j"): 0.05,
    ("-a", "-j"): 0.95,
}
CPT_M = {  # Variables: ["A", "M"]
    ("+a", "+m"): 0.7,
    ("+a", "-m"): 0.3,
    ("-a", "+m"): 0.01,
    ("-a", "-m"): 0.99,
}

# Associate each factor with its local variable mapping
FACTORS = [
    (CPT_B, ["B"]),
    (CPT_E, ["E"]),
    (CPT_A, ["B", "E", "A"]),
    (CPT_J, ["A", "J"]),
    (CPT_M, ["A", "M"]),
]

def apply_evidence(factor, variables, evidence):
    """
    Apply evidence to a factor based on its local variable order.
    """
    restricted_factor = {}
    for assignment, prob in factor.items():
        match = True
        for var, val in evidence.items():
            if var in variables:
                var_index = variables.index(var)  # Get local index of the variable
                if assignment[var_index] != val:
                    match = False
                    break
        if match:
            restricted_factor[assignment] = prob
    return restricted_factor


def normalize(factor):
    """
    Normalize a factor so that the probabilities sum to 1.
    """
    total = sum(factor.values())
    if total == 0 or not factor:  # Check if factor is empty
        print("Normalization error: Factor is empty or total probability is zero.")
        return {}
    return {assignment: prob / total for assignment, prob in factor.items()}

def consistent_assignments(assignment1, assignment2, vars1, vars2):
    """
    Check if two assignments are consistent based on variable names.
    """
    for var in set(vars1) & set(vars2):  # Overlapping variables
        index1 = vars1.index(var)
        index2 = vars2.index(var)
        if assignment1[index1] != assignment2[index2]:
            return False
    return True

def merge_assignments(assignment1, assignment2):
    """
    Merge two consistent assignments.
    """
    return assignment1 + assignment2[len(assignment1):]

def marginalize(factor, variable_index):
    """
    Marginalize out a variable from a factor.
    """
    marginalized_factor = {}
    for assignment, prob in factor.items():
        reduced_assignment = tuple(v for i, v in enumerate(assignment) if i != variable_index)
        marginalized_factor[reduced_assignment] = marginalized_factor.get(reduced_assignment, 0) + prob
    return marginalized_factor

def multiply_factors(factor1, vars1, factor2, vars2):
    """
    Multiply two factors, considering all overlapping variables.
    """
    product = {}
    product_vars = list(vars1) + [var for var in vars2 if var not in vars1]  # Combine variable lists

    for assignment1, prob1 in factor1.items():
        for assignment2, prob2 in factor2.items():
            # Check consistency for overlapping variables
            if consistent_assignments(assignment1, assignment2, vars1, vars2):
                # Merge assignments to create a new assignment
                merged_assignment = tuple(
                    assignment1[vars1.index(var)] if var in vars1 else assignment2[vars2.index(var)]
                    for var in product_vars
                )
                product[merged_assignment] = product.get(merged_assignment, 0) + prob1 * prob2

    if not product:
        print("Warning: Multiplying factors resulted in an empty factor.")
    return product, product_vars

def combine_factors_with_hidden_variable(hidden_var, factors_with_var):
    """
    Combine all factors that include the hidden variable.
    """
    combined_factor, combined_vars = factors_with_var[0]
    for factor, vars_ in factors_with_var[1:]:
        combined_factor, combined_vars = multiply_factors(combined_factor, combined_vars, factor, vars_)

    return combined_factor, combined_vars


def variable_elimination(query_var, evidence, hidden_vars):
    """
    Perform variable elimination with local variable mappings for each factor.
    """
    factors = FACTORS

    # Step 1: Apply evidence
    updated_factors = []
    for factor, variables in factors:
        restricted_factor = apply_evidence(factor, variables, evidence)
        updated_factors.append((restricted_factor, variables))

    # Step 2: Eliminate hidden variables
    for hidden_var in hidden_vars:
        factors_with_var = []
        factors_without_var = []

        # Separate factors with and without the hidden variable
        for factor, variables in updated_factors:
            if hidden_var in variables:
                factors_with_var.append((factor, variables))
            else:
                factors_without_var.append((factor, variables))


        # Combine all factors containing the hidden variable
        combined_factor, combined_vars = combine_factors_with_hidden_variable(hidden_var, factors_with_var)

        # Marginalize out the hidden variable
        marginalized = marginalize(combined_factor, combined_vars.index(hidden_var))
        combined_vars.remove(hidden_var)

        # Add the marginalized factor back to the list
        factors_without_var.append((marginalized, combined_vars))
        updated_factors = factors_without_var

    # Step 3: Multiply remaining factors
    final_factor, final_vars = updated_factors[0]
    for next_factor, next_vars in updated_factors[1:]:
        final_factor, final_vars = multiply_factors(final_factor, final_vars, next_factor, next_vars)

    return normalize(final_factor)

def print_probability_table(factor, query_var):
    """
    Print the probability distribution table in a formatted way.
    """
    print(f"\nProbability Distribution Table for {query_var} (Given Evidence):")
    print("-" * 40)
    print(f"| {query_var:^10} | Probability     |")
    print("-" * 40)
    for assignment, prob in factor.items():
        print(f"| {assignment[0]:^10} | {prob:.12f} |")
    print("-" * 40)


# Main function
if __name__ == "__main__":
    query_var = "B"  # Query variable
    evidence = {"J": "+j"}  # Evidence: John calls
    hidden_vars = ["E", "A", "M"]  # Hidden variables to eliminate

    # Perform variable elimination
    result = variable_elimination(query_var, evidence, hidden_vars)
    
    # Print the probability distribution table
    print_probability_table(result, query_var)

    # Print the result for confirmation
    print("\nP(Burglary | John Calls = +j):", result)
