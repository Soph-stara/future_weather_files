# base_morpher.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

class BaseMorpher(ABC):
    """Abstract base class for EPW variable morphing"""
    
    def __init__(self, epw_column: int, variable_name: str):
        self.epw_column = epw_column  # Column index in EPW file
        self.variable_name = variable_name
        
    @abstractmethod
    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path], 
                      experiment: str, target_year: str) -> List[float]:
        """
        Implement specific morphing algorithm for the variable
        
        Args:
            base_data: Original EPW data for this variable
            climate_files: Dictionary of relevant climate files
            experiment: Selected climate experiment (e.g., 'A2a')
            target_year: Target year for morphing
            
        Returns:
            List of morphed hourly values
        """
        pass
