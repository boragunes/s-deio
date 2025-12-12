import pandas as pd

def calculate_median_rmse(file_name):
    """
    Reads a CSV file, calculates the median RMSE for each unique
    combination of 'Sequence' and 'Config' (e.g., 'spot_forest_hard_default').

    Args:
        file_name (str): The path to the input CSV file.

    Returns:
        pandas.Series or str: A Series with the combined names as the index
                              and median RMSE as the values, or an error message.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_name)

        # 1. Create a new column by concatenating 'Sequence' and 'Config'
        # This creates the desired group names, e.g., 'spot_forest_hard_default'
        df['Combined_Group'] = df['Sequence'] + '_' + df['Config']

        # 2. Group by the new 'Combined_Group' column and calculate the median of 'RMSE (m)'
        median_rmse = df.groupby('Combined_Group')['RMSE (m)'].median()

        return median_rmse

    except FileNotFoundError:
        return f"Error: The file '{file_name}' was not found."
    except KeyError as e:
        return f"Error: A required column was not found. Please ensure the CSV has 'Sequence', 'Config', and 'RMSE (m)' columns. Details: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == '__main__':
    # Use the mock data file created above for demonstration
    file_name = 'evo_results_new_structure.csv' 
    result = calculate_median_rmse(file_name)

    print("Median RMSE per Sequence and Config combination:")
    print("-" * 50)
    print(result)
