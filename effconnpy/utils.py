import numpy as np
import pandas as pd
from typing import Union, List

def validate_input(data: Union[np.ndarray, pd.DataFrame, List[np.ndarray]]) -> np.ndarray:
    """
    Validate and convert input to numpy array
    
    Args:
        data: Input time series data
    
    Returns:
        Numpy array of time series
    """
    if isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError("Input must be numpy array, pandas DataFrame, or list of arrays")

