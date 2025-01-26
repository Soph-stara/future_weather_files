# dew_point_temp_morpher.py

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class DewPointMorpher(BaseMorpher):
    """Implementation of dew point temperature calculation following ASHRAE Handbook equations."""
    
    def __init__(self):
        super().__init__(epw_column=7, variable_name='DPT')
        self.required_files = []  # No dif files needed
        
        # Constants from ASHRAE handbook
        self.C14 = 6.54
        self.C15 = 14.526
        self.C16 = 0.7389
        self.C17 = 0.09486
        self.C18 = 0.4569

    def calculate_saturation_pressure(self, dry_bulb_temp):
        """
        Calculate saturation pressure of water vapour (p'_ws)
        using values from ASHRAE Handbook Chapter 6, Table 3
        """
        a = 17.27
        b = 237.7
        return 0.611 * np.exp((a * dry_bulb_temp) / (b + dry_bulb_temp))

    def calculate_partial_pressure(self, future_rh, saturation_pressure):
        """
        Calculate future partial pressure of water vapour:
        p_w = Φ · p'_ws|dbt,pat
        """
        return (future_rh / 100.0) * saturation_pressure

    def calculate_dew_point(self, pw):
        """
        Calculate dew point temperature using ASHRAE equations:
        
        For 0°C ≤ dpt ≤ 93°C:
        dpt = C14 + C15·α + C16·α² + C17·α³ + C18·(pw)^0.1984
        
        For dpt < 0°C:
        dpt = 6.09 + 12.608·α + 0.4959·α²
        """
        alpha = np.log(pw)
        
        # First try equation for dpt between 0 and 93°C
        dpt = (self.C14 + self.C15 * alpha + 
               self.C16 * alpha**2 + 
               self.C17 * alpha**3 + 
               self.C18 * pw**0.1984)
        
        # If dpt < 0°C, use alternative equation
        if isinstance(dpt, np.ndarray):
            mask = dpt < 0
            dpt[mask] = 6.09 + 12.608 * alpha[mask] + 0.4959 * alpha[mask]**2
        elif dpt < 0:
            dpt = 6.09 + 12.608 * alpha + 0.4959 * alpha**2
            
        return dpt


    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                    experiment: str, target_year: str, 
                    future_temp: pd.Series = None,
                    future_rh: pd.Series = None) -> List[float]:
        """
        Calculate future dew point temperature following ASHRAE procedure.
        """
        try:
            if future_temp is None or future_rh is None:
                raise ValueError("Both future temperature and future RH are required")
                
            print("\nProcessing dew point temperature calculation...")
            print(f"Number of hours: {len(base_data)}")
            
            morphed_dpt = []
            
            # Calculate future dew point for each hour
            for temp, rh in zip(future_temp, future_rh):
                # Step 1: Calculate p'_ws at the future temperature
                pws = self.calculate_saturation_pressure(temp)
                
                # Step 2: Calculate p_w = Φ · p'_ws using future RH
                pw = self.calculate_partial_pressure(rh, pws)
                
                # Step 3: Calculate dew point temperature
                future_dpt = self.calculate_dew_point(pw)
                morphed_dpt.append(float(future_dpt))  # Convert np.float64 to float
            
            # Convert to pandas Series for consistent handling
            morphed_series = pd.Series(morphed_dpt)
            original_dpt = base_data
            
            print("\nDew Point Temperature calculation results:")
            print(f"Original DPT range: {original_dpt.min():.1f}°C to {original_dpt.max():.1f}°C")
            print(f"New DPT range: {morphed_series.min():.1f}°C to {morphed_series.max():.1f}°C")
            print(f"Average DPT change: {(morphed_series.mean() - original_dpt.mean()):.1f}°C")
            
            # Sample verification
            print("\nSample data verification (first 5 hours):")
            print("Original DPT:", [f"{x:.1f}" for x in original_dpt[:5].tolist()])
            print("Future DPT:", [f"{x:.1f}" for x in morphed_series[:5].tolist()])
            print(f"Original mean: {original_dpt.mean():.2f}°C")
            print(f"Future mean: {morphed_series.mean():.2f}°C")
            print(f"Relative change: {((morphed_series.mean() - original_dpt.mean()) / abs(original_dpt.mean()) * 100):+.1f}%")
            
            return morphed_dpt
            
        except Exception as e:
            print(f"Error in dew point temperature calculation: {e}")
            import traceback
            traceback.print_exc()
            return None