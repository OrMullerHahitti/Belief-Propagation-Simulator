from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


# plot results:
def plot_results_from_csv(csv: Path | str = "tests/results/costs_test_1.csv"):
    """
    Plot the results from a CSV file.

    Args:
        csv (Path|str): Path to the CSV file containing the results.
    """
    # Read the CSV file
    data = pd.read_csv(csv)

    # Extract the relevant columns
    x = data.iloc[:, 0]  # Assuming the first column is x-axis data
    y = data.iloc[:, 1]  # Assuming the second column is y-axis data

    # Create a scatter plot
    plt.scatter(x, y)

    # Add labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of Results")

    # Show the plot
    plt.show()


plot_results_from_csv()
