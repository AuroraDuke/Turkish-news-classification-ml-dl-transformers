import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def category_pie_chart(df, column_name):
    """
    Plots a pie chart for the specified column in the DataFrame and prints category ratios.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column to analyze.

    Returns:
        None
    """
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the DataFrame.")
        return

    # Count occurrences of each category
    category_counts = df[column_name].value_counts()

    # Plot as a pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    category_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90, colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'])

    # Add title
    ax.set_title(f'{column_name} Distribution', fontsize=14)
    ax.set_ylabel('')  # Remove the default y-label

    # Show the plot
    plt.show()

    # Calculate class distribution with Counter
    class_distribution = Counter(df[column_name])

    # Total number of data points
    total_count = sum(class_distribution.values())

    # Calculate and display percentage distribution
    print("Original data class distribution:")
    for label, count in class_distribution.items():
        percentage = (count / total_count) * 100
        print(f"{label}: {count} instances ({percentage:.2f}%)")

# Example usage
# df_main = pd.DataFrame({'category': ['A', 'B', 'A', 'C', 'A', 'B', 'C', 'B', 'C', 'A']})
# category_pie_chart(df_main, 'category')