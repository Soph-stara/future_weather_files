# morphers/dry_bulb_temp_morpher.py

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class TemperatureMorpher(BaseMorpher):
    """Implementation of temperature morphing algorithm"""
    
    def __init__(self):
        super().__init__(epw_column=6, variable_name='TEMP')
        # Specify which HADCM3 files are needed for this morpher
        self.required_files = ['TEMP', 'TMAX', 'TMIN']
    
    def calculate_monthly_statistics(self, hourly_data: pd.Series) -> Dict:
        """Calculate monthly statistics for dry bulb temperature"""
        monthly_data = {}
        hours_per_month = 24 * np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        start_idx = 0
        
        for month in range(12):
            end_idx = start_idx + hours_per_month[month]
            month_data = hourly_data[start_idx:end_idx]
            
            monthly_data[month + 1] = {
                'mean': month_data.mean(),
                'max': month_data.max(),
                'min': month_data.min()
            }
            
            start_idx = end_idx
            
        return monthly_data

    def read_dif_file(self, file_path: Path) -> Optional[List[float]]:
        """
        Read a HADCM3 .dif file and return monthly values.
        Returns None if there is an error reading the file.
        """
        try:
            monthly_values = []
            current_month = None
            month_value = None
            time_period = None
            variable_type = None
            
            print(f"\nReading file: {file_path.name}")
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    # Check for time period information
                    if "Mean Change values for" in line:
                        time_period = line.strip()
                        print(f"Time period: {time_period}")
                        continue

                    # Check for variable type
                    if any(var in line for var in ["Mean daily", "Maximum daily", "Minimum daily"]):
                        variable_type = line.strip()
                        print(f"Variable type: {variable_type}")
                        continue
                        
                    if "Month is" in line:
                        if month_value is not None and current_month is not None:
                            monthly_values.append(month_value)
                            print(f"Month {current_month}: {month_value:.2f}°C")
                        current_month = line.split("Month is")[1].strip()
                        month_value = None
                        continue
                    
                    # Skip header lines
                    if any(header in line for header in [
                        "IPCC", "Grid is", "Mean Change", "HADCM", 
                        "Format is", "Maximum", "Minimum", "format is", 
                        "missing code"
                    ]):
                        continue
                        
                    try:
                        # Split line into 8-character chunks and convert to float
                        chunks = [line[i:i+8] for i in range(0, len(line), 8)]
                        numbers = [float(chunk) for chunk in chunks if chunk.strip() and float(chunk) < 9999]
                        
                        if numbers and month_value is None:
                            month_value = numbers[0]
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line.strip()} - {str(e)}")
                        continue
            
            # Don't forget the last month
            if month_value is not None:
                monthly_values.append(month_value)
                print(f"Month {current_month}: {month_value:.2f}°C")
                
            if len(monthly_values) >= 12:
                print(f"\nSuccessfully read {len(monthly_values)} monthly values")
                if len(monthly_values) > 12:
                    print(f"Warning: Found {len(monthly_values)} values, using first 12")
                return monthly_values[:12]
            else:
                raise ValueError(f"Only found {len(monthly_values)} monthly values in {file_path}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                      experiment: str, target_year: str) -> Optional[List[float]]:
        """
        Implement temperature-specific morphing algorithm using:
        dbt = dbt₀ + ΔTEMPₘ + αdbtₘ · (dbt₀ - ⟨dbt₀⟩ₘ)
        where:
        dbt = Future temperature
        dbt₀ = Present-day temperature
        ΔTEMPₘ = Predicted change in monthly mean temperature
        αdbtₘ = Scaling factor for temperature variation
        ⟨dbt₀⟩ₘ = Monthly mean of present-day temperature
        """
        try:
            print("\nProcessing temperature morphing...")
            print(f"EPW column index: {self.epw_column}")
            
            # Calculate monthly statistics for base temperature
            print("\nCalculating base temperature statistics...")
            monthly_stats = self.calculate_monthly_statistics(base_data)
            print(f"Original temperature range: {base_data.min():.1f}°C to {base_data.max():.1f}°C")
            print(f"Original mean temperature: {base_data.mean():.1f}°C")
            
            # Read temperature change files
            delta_tmax = self.read_dif_file(climate_files['TMAX'][0])
            delta_tmin = self.read_dif_file(climate_files['TMIN'][0])
            delta_temp = self.read_dif_file(climate_files['TEMP'][0])
            
            if not all([delta_tmax, delta_tmin, delta_temp]):
                print("Error: Failed to read one or more temperature difference files")
                return None
            
            # Calculate scaling factors (αdbtₘ) for each month
            print("\nCalculating monthly scaling factors...")
            scaling_factors = []
            for month in range(12):
                dbt0_max = monthly_stats[month + 1]['max']
                dbt0_min = monthly_stats[month + 1]['min']
                
                try:
                    alpha = (delta_tmax[month] - delta_tmin[month]) / (dbt0_max - dbt0_min)
                    scaling_factors.append(alpha)
                    print(f"Month {month + 1}: {alpha:.3f}")
                except ZeroDivisionError:
                    print(f"Warning: Could not calculate scaling factor for month {month+1}")
                    scaling_factors.append(1.0)
            
            # Apply morphing equation to each hour
            print("\nApplying morphing equation...")
            morphed_temps = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, dbt0 in enumerate(base_data):
                # Check if we need to move to next month
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Get monthly mean temperature
                dbt0_mean = monthly_stats[current_month]['mean']
                
                # Apply morphing equation
                alpha = scaling_factors[current_month - 1]
                delta_t = delta_temp[current_month - 1]
                
                morphed_temp = dbt0 + delta_t + alpha * (dbt0 - dbt0_mean)
                morphed_temps.append(morphed_temp)
                
                month_hour_count += 1
            
            # Print verification statistics
            # Format statistics
            morphed_array = np.array(morphed_temps)
            mean_change = np.mean(morphed_array) - base_data.mean()
            
            print("\nTemperature morphing statistics:")
            print(f"Morphed temperature range: {np.min(morphed_array):.1f}°C to {np.max(morphed_array):.1f}°C")
            print(f"Morphed mean temperature: {np.mean(morphed_array):.1f}°C")
            print(f"Average temperature change: {mean_change:.1f}°C")
            
            print("\nSample data verification:")
            print(f"Original first 5: {base_data[:5].tolist()}")
            # Format morphed values to 1 decimal place
            morphed_first_5 = [f"{temp:.1f}" for temp in morphed_array[:5]]
            print(f"Morphed first 5:  {morphed_first_5}")
            print(f"Original mean: {base_data.mean():.2f}°C")
            print(f"Morphed mean:  {np.mean(morphed_array):.2f}°C")
            print(f"Change: {mean_change:.1f}°C ({(mean_change / base_data.mean() * 100):+.1f}%)")
            
            return morphed_temps
            
        except Exception as e:
            print(f"Error in temperature morphing: {e}")
            import traceback
            traceback.print_exc()
            return None