# skyline_matcher.py
import numpy as np
from typing import List, Tuple, Optional
# from skyline_database import SkylineDatabase # Assumes SkylineDatabase is in a separate file

class SkylineMatcher:
    """
    Compares a query skyline vector against a database of skyline vectors
    to find the best matches.
    """
    def __init__(self, metric: str = 'mse'):
        """
        Initializes the SkylineMatcher.

        Args:
            metric (str): The distance metric to use for comparison.
                        Supported: 'mse' (Mean Squared Error), 'mae' (Mean Absolute Error).
                        Future: 'dtw', 'cosine'.
        """
        if metric not in ['mse', 'mae']:
            raise ValueError(f"Unsupported metric: {metric}. Supported metrics are 'mse', 'mae'.")
        self.metric = metric

    def _calculate_distance(self, skyline1: np.ndarray, skyline2: np.ndarray) -> float:
        """
        Calculates the distance between two skyline vectors.
        Assumes skyline vectors are pre-processed to be comparable (e.g., same length).
        If not, this method or the calling method needs to handle resizing/interpolation.

        Args:
            skyline1 (np.ndarray): The first skyline vector.
            skyline2 (np.ndarray): The second skyline vector.

        Returns:
            float: The calculated distance.
        """
        if skyline1.shape != skyline2.shape:
            print(f"Warning: Skyline vectors have different lengths ({skyline1.shape[0]} vs {skyline2.shape[0]}). "
                "Results may be unreliable. Consider resampling to a common length.")
            len1, len2 = len(skyline1), len(skyline2)
            if len1 < len2:
                skyline1 = np.pad(skyline1, (0, len2 - len1), 'edge') # Pad with edge values
            elif len2 < len1:
                skyline2 = np.pad(skyline2, (0, len1 - len2), 'edge')

        if self.metric == 'mse':
            return float(np.mean((skyline1 - skyline2)**2))
        elif self.metric == 'mae':
            return float(np.mean(np.abs(skyline1 - skyline2)))
        # Add other metrics like DTW or cosine similarity here

        return float('inf') # Should not happen if metric is validated

    def find_matches(self, 
                    query_skyline: np.ndarray, 
                    database_skylines: List[Tuple[str, np.ndarray]], 
                    top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Finds the top N most similar skylines from the database to the query skyline.

        Args:
            query_skyline (np.ndarray): The skyline vector of the query image.
            database_skylines (List[Tuple[str, np.ndarray]]): A list of tuples,
                where each tuple is (image_id, skyline_vector).
            top_n (int): The number of top matches to return.

        Returns:
            List[Tuple[str, float]]: A list of (image_id, distance_score) tuples,
                                    sorted by distance (lower is better).
        """
        if not database_skylines:
            print("Database is empty. No matches can be found.")
            return []

        distances = []
        for image_id, db_skyline_vector in database_skylines:
            if db_skyline_vector.ndim == 0 or query_skyline.ndim == 0: 
                print(f"Warning: Skipping invalid skyline vector for ID {image_id} or query.")
                continue

            if db_skyline_vector.ndim > 1: db_skyline_vector = db_skyline_vector.flatten()
            if query_skyline.ndim > 1: query_skyline = query_skyline.flatten()

            dist = self._calculate_distance(query_skyline, db_skyline_vector)
            distances.append((image_id, dist))
    
            distances.sort(key=lambda item: item[1])
        
        return distances[:top_n]