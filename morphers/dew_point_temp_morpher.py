from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class DewPointMorpher(BaseMorpher):
    """Implementation of dew point temperature calculation with GeoTIFF support."""
    
    def __init__(self):
        super().__init__(epw_column=7, variable_name='DPT')
        self.required_files = ['TEMP_historic', 'VAPR_historic', 'TMAX_future', 'TMIN_future']
        
        # ASHRAE constants for dew point calculation
        self.C14 = 6.54
        self.C15 = 14.526
        self.C16 = 0.7389
        self.C17 = 0.09486
        self.C18 = 0.4569

    def calculate_saturation_pressure(self, dry_bulb_temp):
        """Calculate saturation pressure of water vapour"""
        a = 17.27
        b = 237.7
        return 0.611 * np.exp((a * dry_bulb_temp) / (b + dry_bulb_temp))

    def calculate_partial_pressure(self, rh, saturation_pressure):
        """Calculate partial pressure of water vapour"""
        return (rh / 100.0) * saturation_pressure

    def calculate_dew_point(self, pw):
        """Calculate dew point temperature"""
        alpha = np.log(pw)
        
        dpt = (self.C14 + self.C15 * alpha + 
               self.C16 * alpha**2 + 
               self.C17 * alpha**3 + 
               self.C18 * pw**0.1984)
        
        if isinstance(dpt, np.ndarray):
            mask = dpt < 0
            dpt[mask] = 6.09 + 12.608 * alpha[mask] + 0.4959 * alpha[mask]**2
        elif dpt < 0:
            dpt = 6.09 + 12.608 * alpha + 0.4959 * alpha**2
            
        return dpt

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                  scenario: str, period: str, future_temp: pd.Series = None,
                  future_rh: pd.Series = None) -> Optional[List[float]]:
        """Calculate future dew point temperature using GeoTIFF climate data and ASHRAE methodology."""
        try:
            print("\nProcessing dew point temperature calculation...")
            
            # Check if we have required future data
            if future_temp is None or future_rh is None:
                print("Error: Future temperature and RH data required for dew point calculation")
                return None
                
            # Get location using base class method
            location_info = self.get_location_from_epw()
            if not location_info:
                return None
            lat, lon, city = location_info
            
            # Calculate future dew points using ASHRAE methodology
            morphed_dpt = []
            for hour in range(len(base_data)):
                try:
                    # 1. Calculate saturation pressure at future temperature
                    p_ws = self.calculate_saturation_pressure(future_temp[hour])
                    
                    # 2. Calculate future partial pressure using future RH
                    pw = self.calculate_partial_pressure(future_rh[hour], p_ws)
                    
                    # 3. Calculate dew point using ASHRAE formula
                    dpt = self.calculate_dew_point(pw)
                    
                    # Validate dew point temperature
                    if not self.validate_value(dpt, 'TEMP', city):
                        print(f"Warning: Calculated dew point {dpt:.2f}°C at hour {hour} outside expected range for {city}")
                    
                    # Ensure dew point doesn't exceed dry bulb temperature
                    if dpt > future_temp[hour]:
                        print(f"Warning: Adjusting dew point at hour {hour} to match dry bulb temperature")
                        dpt = future_temp[hour]
                    
                    morphed_dpt.append(float(dpt))
                    
                except Exception as e:
                    print(f"Error calculating dew point for hour {hour}: {e}")
                    # Use original dew point as fallback
                    morphed_dpt.append(float(base_data[hour]))
            
            # Statistics and reporting
            if morphed_dpt:
                morphed_series = pd.Series(morphed_dpt)
                print("\nDew Point Temperature calculation results:")
                print(f"Original DPT range: {base_data.min():.1f}°C to {base_data.max():.1f}°C")
                print(f"New DPT range: {morphed_series.min():.1f}°C to {morphed_series.max():.1f}°C")
                print(f"Average DPT change: {(morphed_series.mean() - base_data.mean()):.1f}°C")
                
                # Sample verification
                print("\nSample verification (first 5 values):")
                for i in range(min(5, len(morphed_dpt))):
                    print(f"Hour {i}:")
                    print(f"  Dry Bulb: {future_temp[i]:.1f}°C")
                    print(f"  RH: {future_rh[i]:.1f}%")
                    print(f"  Dew Point: {morphed_dpt[i]:.1f}°C")
                
                return morphed_dpt
            else:
                print("Error: No dew points calculated")
                return None
                
        except Exception as e:
            print(f"Error in dew point temperature calculation: {e}")
            import traceback
            traceback.print_exc()
            return None