import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    
    """
    Reads a CSV file and loads it into a pandas DataFrame.

        Parameters
        ----------
        file_path : str
            The path to the CSV file.

        Returns
        -------
        pd.DataFrame
            The DataFrame loaded from the CSV file.
    """

    return pd.read_csv(file_path, delimiter="\t")