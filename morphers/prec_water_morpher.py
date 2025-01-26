# morphers/prec_water_morpher.py

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class PrecipitableWaterMorpher(BaseMorpher):
    """Implementation of precipitable water morphing algorithm using HadCM3 data"""
    
    def __init__(self):
        # Precipitable water is in column 28 (0-based index)
        super().__init__(epw_column=28, variable_name='PREC')
        self.required_files = ['PREC']
        
    def detect_epw_format(self, data: pd.Series, filepath: str) -> str:
        """
        Detect whether the EPW file is IWEC or TMYx format
        
        Args:
            data: The precipitable water data series
            filepath: Path to the EPW file
        
        Returns:
            str: 'IWEC' or 'TMYx'
        """
        # Check filename first
        if 'IWEC' in str(filepath):
            return 'IWEC'
        if 'TMYx' in str(filepath):
            return 'TMYx'
        
        # If filename doesn't help, check data characteristics
        mean_value = data.mean()
        if mean_value < 0.1:  # IWEC typically uses smaller values
            return 'IWEC'
        return 'TMYx'
    
    def preprocess_base_data(self, data: pd.Series, filepath: str) -> pd.Series:
        """
        Preprocess the base data to handle different EPW formats
        
        Args:
            data: Original precipitable water data
            filepath: Path to the EPW file
        
        Returns:
            pd.Series: Processed data in consistent units
        """
        format_type = self.detect_epw_format(data, filepath)
        print(f"\nDetected EPW format: {format_type}")
        print(f"Original data statistics:")
        print(f"Mean: {data.mean():.3f}")
        print(f"Min: {data.min():.3f}")
        print(f"Max: {data.max():.3f}")
        
        if format_type == 'IWEC':
            # Convert IWEC units to match TMYx
            converted_data = data * 10.0  # Conversion factor for IWEC to TMYx units
            print(f"\nConverted IWEC data statistics:")
            print(f"Mean: {converted_data.mean():.3f}")
            print(f"Min: {converted_data.min():.3f}")
            print(f"Max: {converted_data.max():.3f}")
            return converted_data
        
        return data
    
    def calculate_monthly_statistics(self, hourly_data: pd.Series) -> Dict:
        """Calculate monthly statistics for precipitable water"""
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

    def read_dif_file(self, file_path: Path) -> Optional[List[float]]:
        """
    Read a HADCM3 .dif file and return monthly values for precipitation change.
    Handles all valid values including zeros for accurate precipitation changes.
    
    Args:
        file_path: Path to the .dif file
            
    Returns:
        Optional[List[float]]: List of 12 monthly change values, or None if reading fails
        """
        try:
            monthly_values = []
            current_month_data = []
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                print(f"\nReading precipitation change file: {file_path}")
                
                # Print header information for verification
                header_lines = [line.strip() for line in lines[:5]]
                for line in header_lines:
                    print(line)
            
            current_month = None
            for line in lines:
                # Check for month marker and process previous month's data
                if "Month is" in line:
                    month = line.split("Month is")[1].strip()
                    
                    # Process previous month's data if exists
                    if current_month_data:
                        # Include all valid values (including zeros)
                        valid_values = [x for x in current_month_data if x < 9999]
                        if valid_values:
                            month_value = sum(valid_values) / len(valid_values)
                            monthly_values.append(month_value)
                            print(f"Month {current_month}: Found {len(valid_values)} valid values, "
                                f"average change: {month_value:.6f} mm/day")
                        else:
                            monthly_values.append(0.0)
                            print(f"Month {current_month}: No valid values found, using 0.0")
                    
                    current_month = month
                    current_month_data = []
                    continue
                
                # Skip header lines
                if any(header in line for header in [
                    "IPCC", "Grid is", "Mean Change", "HADCM", 
                    "Format is", "missing code", "Total Precipitation"
                ]):
                    continue
                    
                try:
                    # Split line into 8-character chunks and convert to floats
                    chunks = [line[i:i+8] for i in range(0, len(line), 8)]
                    numbers = [float(chunk) for chunk in chunks if chunk.strip()]
                    
                    # Add all valid numbers to current month's data
                    valid_numbers = [x for x in numbers if x < 9999]
                    current_month_data.extend(valid_numbers)
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line: {line.strip()} - {str(e)}")
                    continue
            
            # Process the last month
            if current_month_data:
                valid_values = [x for x in current_month_data if x < 9999]
                if valid_values:
                    month_value = sum(valid_values) / len(valid_values)
                    monthly_values.append(month_value)
                    print(f"Month {current_month}: Found {len(valid_values)} valid values, "
                        f"average change: {month_value:.6f} mm/day")
                else:
                    monthly_values.append(0.0)
                    print(f"Month {current_month}: No valid values found, using 0.0")
            
            # Verify we have the correct number of months
            if len(monthly_values) >= 12:
                if len(monthly_values) > 12:
                    print(f"Warning: Found {len(monthly_values)} months, using first 12")
                monthly_values = monthly_values[:12]
                
                print("\nMonthly precipitation changes (mm/day):")
                for month, value in enumerate(monthly_values, 1):
                    print(f"Month {month}: {value:.6f}")
                
                return monthly_values
            else:
                raise ValueError(f"Only found {len(monthly_values)} monthly values in {file_path}")
                    
        except Exception as e:
            print(f"Error reading precipitation file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_scaling_factors(self, delta_prec: List[float], base_mean: float) -> List[float]:
        """Calculate scaling factors for precipitation"""
        scaling_factors = []
        for change in delta_prec:
            factor = 1 + (change / base_mean) if base_mean > 0 else 1
            scaling_factors.append(factor)
        return scaling_factors

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                    experiment: str, target_year: str) -> Optional[List[float]]:
        try:
            print("\nStarting precipitation morphing...")
            
            # Preprocess the base data
            processed_data = self.preprocess_base_data(base_data, str(climate_files['PREC'][0]))
            base_mean = processed_data.mean()
            
            # Read precipitation change data
            delta_prec = self.read_dif_file(climate_files['PREC'][0])
            if delta_prec is None:
                print("Error: Failed to read precipitation difference file")
                return None
            
            # Calculate scaling factors
            scaling_factors = self.calculate_scaling_factors(delta_prec, base_mean)
            
            print("\nMonthly scaling factors:")
            for month, factor in enumerate(scaling_factors, 1):
                print(f"Month {month}: {factor:.6f} ({(factor-1)*100:+.2f}%)")
            
            # Apply morphing equation
            morphed_prec = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                            31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, precip in enumerate(processed_data):
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Apply scaling factor
                alpha = scaling_factors[current_month - 1]
                morphed_value = alpha * precip
                
                morphed_prec.append(morphed_value)
                month_hour_count += 1
            
            # Create numpy array for calculations
            morphed_array = np.array(morphed_prec)
            
            # Print verification statistics
            print("\nVerification statistics:")
            print(f"Original range: {processed_data.min():.3f} to {processed_data.max():.3f} mm")
            print(f"Morphed range: {morphed_array.min():.3f} to {morphed_array.max():.3f} mm")
            print(f"Original mean: {processed_data.mean():.3f} mm")
            print(f"Morphed mean: {morphed_array.mean():.3f} mm")
            abs_change = morphed_array.mean() - processed_data.mean()
            rel_change = (abs_change / processed_data.mean()) * 100
            print(f"Change: {abs_change:+.3f} mm ({rel_change:+.1f}%)")
            
            return morphed_prec
                
        except Exception as e:
            print(f"Error in precipitation morphing: {e}")
            import traceback
            traceback.print_exc()
            return None