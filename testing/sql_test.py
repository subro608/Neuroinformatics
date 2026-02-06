import os
import sqlite3

import pandas as pd


def get_best_trial_parameters(db_path):
    """
    Extract the best trial and its parameters from the Optuna database

    Args:
        db_path (str): Path to the Optuna SQLite database

    Returns:
        dict: Best parameters and their values
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return None

    conn = sqlite3.connect(db_path)

    # First, find the best trial (lowest value since direction is MINIMIZE)
    best_trial_query = """
    SELECT 
        t.trial_id,
        t.number,
        tv.value as objective_value
    FROM 
        trials t
    JOIN 
        trial_values tv ON t.trial_id = tv.trial_id
    WHERE 
        t.state = 'COMPLETE'
    ORDER BY 
        tv.value ASC
    LIMIT 1
    """

    best_trial = pd.read_sql_query(best_trial_query, conn)

    if best_trial.empty:
        conn.close()
        return None

    best_trial_id = best_trial["trial_id"].iloc[0]
    best_trial_number = best_trial["number"].iloc[0]
    best_value = best_trial["objective_value"].iloc[0]

    # Get all parameters for the best trial
    params_query = f"""
    SELECT 
        param_name,
        param_value,
        distribution_json
    FROM 
        trial_params
    WHERE 
        trial_id = {best_trial_id}
    """

    params_df = pd.read_sql_query(params_query, conn)
    conn.close()

    # Create a dictionary of the best parameters
    best_params = {
        "trial_number": best_trial_number,
        "objective_value": best_value,
        "parameters": {},
    }

    for _, row in params_df.iterrows():
        best_params["parameters"][row["param_name"]] = row["param_value"]

    return best_params


def get_top_n_trials(db_path, n=10):
    """
    Extract the top N trials with their parameters

    Args:
        db_path (str): Path to the Optuna SQLite database
        n (int): Number of top trials to extract

    Returns:
        DataFrame: Top N trials with their parameters and values
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return None

    conn = sqlite3.connect(db_path)

    # Get top N trials by objective value
    top_trials_query = f"""
    SELECT 
        t.trial_id,
        t.number,
        tv.value as objective_value
    FROM 
        trials t
    JOIN 
        trial_values tv ON t.trial_id = tv.trial_id
    WHERE 
        t.state = 'COMPLETE'
    ORDER BY 
        tv.value ASC
    LIMIT {n}
    """

    top_trials = pd.read_sql_query(top_trials_query, conn)

    if top_trials.empty:
        conn.close()
        return None

    # For each top trial, get its parameters
    all_top_trials = []

    for _, trial in top_trials.iterrows():
        trial_id = trial["trial_id"]

        params_query = f"""
        SELECT 
            param_name,
            param_value
        FROM 
            trial_params
        WHERE 
            trial_id = {trial_id}
        """

        params = pd.read_sql_query(params_query, conn)

        # Convert parameters to a dictionary
        params_dict = dict(zip(params["param_name"], params["param_value"]))

        # Add trial information
        trial_info = {
            "trial_id": trial_id,
            "trial_number": trial["number"],
            "objective_value": trial["objective_value"],
        }

        # Combine trial info with parameters
        combined = {**trial_info, **params_dict}
        all_top_trials.append(combined)

    conn.close()

    # Convert list of dictionaries to DataFrame
    return pd.DataFrame(all_top_trials)


def get_all_unique_parameters(db_path):
    """
    Get a list of all unique parameter names in the study

    Args:
        db_path (str): Path to the Optuna SQLite database

    Returns:
        list: List of parameter names
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return None

    conn = sqlite3.connect(db_path)

    query = """
    SELECT DISTINCT param_name 
    FROM trial_params
    """

    result = pd.read_sql_query(query, conn)
    conn.close()

    return result["param_name"].tolist()


def get_parameter_distributions(db_path):
    """
    Get the distribution details for all parameters

    Args:
        db_path (str): Path to the Optuna SQLite database

    Returns:
        dict: Dictionary with parameter distributions
    """
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found.")
        return None

    conn = sqlite3.connect(db_path)

    query = """
    SELECT DISTINCT param_name, distribution_json
    FROM trial_params
    """

    distributions = pd.read_sql_query(query, conn)
    conn.close()

    dist_dict = {}
    for _, row in distributions.iterrows():
        dist_dict[row["param_name"]] = row["distribution_json"]

    return dist_dict


# Run the extraction
if __name__ == "__main__":
    db_path = "eegnet.db"

    # Check if file exists
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found. Please verify the file path.")
    else:
        print(f"Found database file: {db_path}")

        # Get all unique parameters
        params = get_all_unique_parameters(db_path)
        print(f"\nParameters being optimized ({len(params)}):")
        for p in params:
            print(f"- {p}")

        # Get best trial
        print("\nBest Trial Parameters:")
        best_params = get_best_trial_parameters(db_path)
        if best_params:
            print(f"Trial Number: {best_params['trial_number']}")
            print(f"Objective Value: {best_params['objective_value']}")
            print("Parameters:")
            for param, value in best_params["parameters"].items():
                print(f"  {param}: {value}")

        # Get top 5 trials
        print("\nTop 5 Trials:")
        top_trials = get_top_n_trials(db_path, 5)
        if top_trials is not None:
            pd.set_option("display.max_columns", None)  # Show all columns
            print(top_trials)

        # Get parameter distributions
        print("\nParameter Distributions:")
        distributions = get_parameter_distributions(db_path)
        if distributions:
            for param, dist in distributions.items():
                print(f"{param}: {dist}")
