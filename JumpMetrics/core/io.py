import logging  # for logging errors
import pandas as pd

def load_cropped_force_data(filepath, freq=None):
    try:
        df = pd.read_csv(filepath, delimiter='\t', skiprows=[0, 2, 3, 4])
    except FileNotFoundError:
        logging.error(f"The file at '{filepath}' was not found.")
        return None
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error while reading the file: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while reading the file: {e}")
        return None
    if freq is None:
        colname = 'FZsum'
    else:
        colname = f'FZsum_{freq}Hz'
    if colname not in df.columns:
        logging.warning(f"Column '{colname}' not found in the dataframe. Available columns: {df.columns.tolist()}")
        return None

    try:
        force_data = df[colname]
    except KeyError:
        logging.warning(f"Key '{colname}' not found in the dataframe.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while accessing the column: {e}")
        return None

    return force_data