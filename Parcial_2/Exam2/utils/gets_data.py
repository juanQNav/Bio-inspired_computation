import pandas as pd
import csv
import numpy as np


def load_knapsack_data(csv_path):
    weights = []
    values = []
    items_df = pd.read_csv(csv_path)
    for _, row in items_df.iterrows():
        weights.append(row['weights'])
        values.append(row['values'])

    return weights, values


def load_iris_data(csv_path):
    df = pd.read_csv(csv_path)

    data = df[['sepal_length', 'sepal_width',
               'petal_length', 'petal_width']].values

    target_mapping = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
    target = df['target'].map(target_mapping).values

    return data, target


def save_results():
    pass
