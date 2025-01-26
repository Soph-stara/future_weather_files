from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class RelativeHumidityMorpher(BaseMorpher):
    """Implementation of relative humidity morphing algorithm"""
    
    def __init__(self):
        # EPW column 8 is relative humidity according to EPWMorphingManager
        super().__init__(epw_column=8, variable_name='RH')
        # Specify which HADCM3 files are needed for this morpher
        self.required_files = ['RHUM']
    
    def calculate_monthly_statistics(self, hourly_data: pd.Series) -> Dict:
        """Calculate monthly statistics for relative humidity"""
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
        """
        Read a HADCM3 .dif file and return monthly values.
        HadCM3 provides relative humidity changes as fractions (e.g., -0.45),
        which need to be converted to percentage points (e.g., -45%).
        """
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
                            # Values are already percentage point changes
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
        Implement relative humidity morphing algorithm using shift function:
        Φ = Φ₀ + ΔRHUMₘ
        where:
        Φ = Future relative humidity (%)
        Φ₀ = Present-day relative humidity (%)
        ΔRHUMₘ = Predicted absolute change of mean relative humidity for month m (%)
        """
        try:
            print("\nProcessing RH...")
            print(f"EPW column index: {self.epw_column}")
            
            # Calculate monthly statistics for base relative humidity
            monthly_stats = self.calculate_monthly_statistics(base_data)
            
            # Read relative humidity change file
            delta_rhum = self.read_dif_file(climate_files['RHUM'][0])
            
            if not delta_rhum:
                print("Error: Failed to read relative humidity difference file")
                return None
            
            # Print original RH statistics
            print("Relative Humidity morphing statistics:")
            print(f"Original RH range: {base_data.min():.1f}% to {base_data.max():.1f}%")
            print(f"Original mean RH: {base_data.mean():.1f}%")
            
            # Print monthly changes
            print("Monthly RH changes (percentage points):")
            for month, delta in enumerate(delta_rhum, 1):
                print(f"Month {month}: {delta:+.3f}%")
            
            # Apply morphing equation to each hour
            morphed_rh = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, rh0 in enumerate(base_data):
                # Check if we need to move to next month
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Apply shift equation: Φ = Φ₀ + ΔRHUMₘ
                delta_rh = delta_rhum[current_month - 1]
                morphed_value = rh0 + delta_rh
                
                # Ensure relative humidity stays within valid range (0-100%)
                morphed_value = max(0, min(100, morphed_value))
                
                morphed_rh.append(morphed_value)
                month_hour_count += 1
            
            # Calculate and print morphed statistics
            print(f"Morphed RH range: {min(morphed_rh):.1f}% to {max(morphed_rh):.1f}%")
            print(f"Morphed mean RH: {np.mean(morphed_rh):.1f}%")
            print(f"Average RH change: {((np.mean(morphed_rh) - base_data.mean()) / base_data.mean() * 100):+.1f}%")
            
            # Print sample data verification
            print("\nSample data verification:")
            print(f"Original first 5: {base_data[:5].tolist()}")
            print(f"Morphed first 5:  {morphed_rh[:5]}")
            print(f"Original mean: {base_data.mean():.2f}")
            print(f"Morphed mean:  {np.mean(morphed_rh):.2f}")
            print(f"Change: {((np.mean(morphed_rh) - base_data.mean()) / base_data.mean() * 100):+.1f}%")
            
            return morphed_rh
            
        except Exception as e:
            print(f"Error in relative humidity morphing: {e}")
            return None