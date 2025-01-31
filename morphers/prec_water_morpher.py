from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class PrecipitableWaterMorpher(BaseMorpher):
    """Implementation of precipitable water morphing algorithm using GeoTIFF data"""
    
    def __init__(self):
        # Precipitable water is in column 28 (0-based index)
        super().__init__(epw_column=28, variable_name='PREC')
        self.required_files = ['PREC_historic', 'PREC_future']
    
    def detect_epw_format(self, data: pd.Series, filepath: str) -> str:
        """Detect whether the EPW file is IWEC or TMYx format"""
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
        """Preprocess the base data to handle different EPW formats"""
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

    def calculate_scaling_factors(self, historic_monthly: List[float], future_monthly: float) -> List[float]:
        """Calculate scaling factors according to αrr_m = 1 + PREC_m/100 formula"""
        scaling_factors = []
        for hist in historic_monthly:
            if hist > 0:
                # Calculate percentage change: PREC_m
                prec_m = ((future_monthly - hist) / hist) * 100
                # Apply formula αrr_m = 1 + PREC_m/100
                factor = 1 + (prec_m/100)
            else:
                factor = 1.0
            scaling_factors.append(factor)
        return scaling_factors

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                    scenario: str, period: str) -> Optional[List[float]]:
        try:
            print("\nStarting precipitation morphing...")
            
            # Get location using base class method
            location_info = self.get_location_from_epw()
            if not location_info:
                return None
            lat, lon, city = location_info
            
            # Get historic monthly precipitation
            print("\nReading historic precipitation data...")
            hist_prec = self.get_monthly_values(climate_files['PREC_historic'], lat, lon)
            if not hist_prec:
                return None
            
            # Get future precipitation
            print("\nReading future precipitation data...")
            future_prec = self.get_value_from_tif(climate_files['PREC_future'][0], lat, lon)
            if future_prec is None:
                return None
                
            print(f"Future precipitation: {future_prec:.2f} mm/day")
            
            # Validate precipitation values
            if not self.validate_value(future_prec, 'PREC', city):
                print(f"Warning: Future precipitation {future_prec:.2f} mm/day outside expected range for {city}")
            
            # Calculate scaling factors
            print("\nCalculating scaling factors...")
            scaling_factors = self.calculate_scaling_factors(hist_prec, future_prec)
            for month, factor in enumerate(scaling_factors, 1):
                print(f"Month {month}: {factor:.3f} ({(factor-1)*100:+.1f}%)")
            
            # Preprocess the base data
            processed_data = self.preprocess_base_data(base_data, str(self.epw_path))
            
            # Apply morphing equation
            print("\nApplying morphing equation...")
            morphed_prec = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                            31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, rr0 in enumerate(processed_data):  # rr0 is the present day EPW value
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Get scaling factor for current month (αrr_m)
                alpha = scaling_factors[current_month - 1]
                
                # Apply the formula: rr = (αrr_m * rr0)
                morphed_value = alpha * rr0
                
                # Validate morphed precipitation
                if not self.validate_value(morphed_value, 'PREC'):
                    print(f"Warning: Morphed precipitation {morphed_value:.2f} mm at hour {hour} outside expected range")
                
                morphed_prec.append(morphed_value)
                month_hour_count += 1
            
            # Print verification statistics
            morphed_array = np.array(morphed_prec)
            print("\nVerification statistics:")
            print(f"Original range: {processed_data.min():.3f} to {processed_data.max():.3f} mm")
            print(f"Morphed range: {morphed_array.min():.3f} to {morphed_array.max():.3f} mm")
            print(f"Original mean: {processed_data.mean():.3f} mm")
            print(f"Morphed mean: {morphed_array.mean():.3f} mm")
            abs_change = morphed_array.mean() - processed_data.mean()
            rel_change = (abs_change / processed_data.mean()) * 100 if processed_data.mean() != 0 else 0
            print(f"Change: {abs_change:+.3f} mm ({rel_change:+.1f}%)")
            
            return morphed_prec
            
        except Exception as e:
            print(f"Error in precipitation morphing: {e}")
            import traceback
            traceback.print_exc()
            return None