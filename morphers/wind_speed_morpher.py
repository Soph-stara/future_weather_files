# morphers/wind_speed_morpher.py

from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class WindSpeedMorpher(BaseMorpher):
    """Implementation of wind speed morphing algorithm using UKCIP02 data"""
    
    def __init__(self):
        # Wind speed is in column 21 of EPW file
        super().__init__(epw_column=21, variable_name='WIND')
        # Specify which HADCM3 files are needed for this morpher
        self.required_files = ['WIND']
        
    def calculate_monthly_statistics(self, hourly_data: pd.Series) -> Dict:
        """Calculate monthly statistics for wind speed"""
        monthly_data = {}
        hours_per_month = 24 * np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        start_idx = 0
        
        for month in range(12):
            end_idx = start_idx + hours_per_month[month]
            month_data = hourly_data[start_idx:end_idx]
            
            monthly_data[month + 1] = {
                'mean': month_data.mean()
            }
            
            start_idx = end_idx
            
        return monthly_data

    def read_dif_file(self, file_path: Path) -> List[float]:
        """Read a HADCM3 .dif file and return monthly values for wind speed change"""
        try:
            monthly_values = []
            current_month = None
            month_value = None
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    if "Month is" in line:
                        if month_value is not None and current_month is not None:
                            monthly_values.append(month_value)
                        current_month = line.split("Month is")[1].strip()
                        month_value = None
                        continue
                    
                    if any(header in line for header in ["IPCC", "Grid is", "Mean Change", "HADCM", "Format is"]):
                        continue
                        
                    try:
                        chunks = [line[i:i+8] for i in range(0, len(line), 8)]
                        numbers = [float(chunk) for chunk in chunks if chunk.strip() and float(chunk) < 9999]
                        if numbers and month_value is None:
                            month_value = numbers[0]
                    except (ValueError, IndexError):
                        continue
            
            if month_value is not None:
                monthly_values.append(month_value)
                
            if len(monthly_values) >= 12:
                return monthly_values[:12]
            else:
                raise ValueError(f"Only found {len(monthly_values)} monthly values in {file_path}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                  experiment: str, target_year: str) -> List[float]:
        """
    Implement wind-specific morphing algorithm using stretch function
    with direct wind speed changes from HADCM3
        """
        try:
            # Log original wind speed statistics
            original_min = base_data.min()
            original_max = base_data.max()
            original_mean = base_data.mean()
            print(f"\nWind Speed morphing statistics:")
            print(f"Original wind speed range: {original_min:.1f} to {original_max:.1f} knots")
            print(f"Original mean wind speed: {original_mean:.1f} knots")

            # Read wind speed change file
            delta_wind = self.read_dif_file(climate_files['WIND'][0])
            
            if not delta_wind:
                print("Error: Failed to read wind speed difference file")
                return None
            
            # Calculate scaling factors using absolute changes
            # Note: delta_wind values are in m/s, need to convert baseline to m/s for scaling
            base_mean_ms = original_mean * 0.514444  # Convert base mean to m/s
            scaling_factors = [1 + (wind_change / base_mean_ms) for wind_change in delta_wind]
            
            # Apply morphing equation to each hour
            morphed_speeds = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                            31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, ws0 in enumerate(base_data):
                # Check if we need to move to next month
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Apply morphing equation: ws = (αwsₘ · ws₀) · 0.514444
                alpha = scaling_factors[current_month - 1]
                morphed_speed = (alpha * ws0) * 0.514444  # Convert to m/s
                
                morphed_speeds.append(morphed_speed)
                month_hour_count += 1
            
            # Convert to numpy array for easier calculations
            morphed_speeds_array = np.array(morphed_speeds)
            
            # Log morphed wind speed statistics
            morphed_min = morphed_speeds_array.min()
            morphed_max = morphed_speeds_array.max()
            morphed_mean = morphed_speeds_array.mean()
            print(f"Morphed wind speed range: {morphed_min:.1f} to {morphed_max:.1f} m/s")
            print(f"Morphed mean wind speed: {morphed_mean:.1f} m/s")
            
            # Calculate and log average change
            original_mean_ms = original_mean * 0.514444  # Convert original mean to m/s
            mean_change = morphed_mean - original_mean_ms
            mean_change_percent = (mean_change / original_mean_ms) * 100
            print(f"Average wind speed change: {mean_change:.2f} m/s ({mean_change_percent:.1f}%)")
            
            # Log the monthly scaling factors
            print("\nMonthly wind speed scaling factors:")
            for month, factor in enumerate(scaling_factors, 1):
                print(f"Month {month}: {factor:.3f}")
            
            return morphed_speeds
            
        except Exception as e:
            print(f"Error in wind speed morphing: {e}")
            return None