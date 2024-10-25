import pandas as pd


def load_knapsack_data(csv_path):
    weights = []
    values = []
    items_df = pd.read_csv(csv_path)
    for _, row in items_df.iterrows():
        weights.append(row['weights'])
        values.append(row['values'])

    return weights, values


def save_results():
    pass
