from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class RelativeHumidityMorpher(BaseMorpher):
    """Implementation of relative humidity morphing algorithm using vapor pressure"""
    
    def __init__(self):
        # EPW column 8 is relative humidity
        super().__init__(epw_column=8, variable_name='RH')
        # Update required files to use available data
        self.required_files = ['TEMP_historic', 'VAPR_historic', 'TMAX_future', 'TMIN_future']
    
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

    def calculate_saturation_vapor_pressure(self, temp_c: float) -> float:
        """Calculate saturation vapor pressure using Magnus formula"""
        # Constants for Magnus formula
        a = 17.27
        b = 237.7  # °C
        
        # Calculate saturation vapor pressure in hPa
        es = 6.112 * np.exp((a * temp_c) / (b + temp_c))
        return es

    def calculate_relative_humidity(self, temp_c: float, vapor_pressure: float) -> float:
        """Calculate relative humidity from temperature and vapor pressure"""
        es = self.calculate_saturation_vapor_pressure(temp_c)
        rh = (vapor_pressure / es) * 100
        return min(100, max(0, rh))  # Constrain to 0-100%

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                      scenario: str, period: str) -> Optional[List[float]]:
        try:
            print("\nProcessing relative humidity...")
            
            # Get location using base class method
            location_info = self.get_location_from_epw()
            if not location_info:
                return None
            lat, lon, city = location_info
            
            # Get historic data with appropriate scaling
            hist_temp = self.get_monthly_values(climate_files['TEMP_historic'], lat, lon, scale_factor=0.1)
            hist_vapr = self.get_monthly_values(climate_files['VAPR_historic'], lat, lon)  # No scaling for vapor pressure
            
            if not all([hist_temp, hist_vapr]):
                return None
            
            # Get future temperature (average of TMAX and TMIN) with scaling
            future_tmax = self.get_value_from_tif(climate_files['TMAX_future'][0], lat, lon, scale_factor=0.1)
            future_tmin = self.get_value_from_tif(climate_files['TMIN_future'][0], lat, lon, scale_factor=0.1)
            
            if future_tmax is None or future_tmin is None:
                return None
            
            future_temp = (future_tmax + future_tmin) / 2
            print(f"Future temperature: {future_temp:.2f}°C")
            
            # Validate temperature values
            if not self.validate_value(future_temp, 'TEMP', city):
                print(f"Warning: Future temperature {future_temp:.2f}°C outside expected range for {city}")
            
            # Calculate change factors
            temp_change = future_temp - np.mean(hist_temp)
            
            # Print original RH statistics
            print("\nRelative Humidity morphing statistics:")
            print(f"Original RH range: {base_data.min():.1f}% to {base_data.max():.1f}%")
            print(f"Original mean RH: {base_data.mean():.1f}%")
            
            # Apply morphing equation to each hour
            print("\nApplying morphing...")
            morphed_rh = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, rh0 in enumerate(base_data):
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Use temperature change to adjust vapor pressure and recalculate RH
                original_es = self.calculate_saturation_vapor_pressure(hist_temp[current_month - 1])
                original_vp = (rh0 / 100.0) * original_es
                
                # Calculate new RH using adjusted temperature
                new_temp = hist_temp[current_month - 1] + temp_change
                morphed_value = self.calculate_relative_humidity(new_temp, original_vp)
                
                # Validate morphed RH
                if not self.validate_value(morphed_value, 'RH'):
                    print(f"Warning: Morphed RH {morphed_value:.1f}% at hour {hour} outside expected range")
                
                morphed_rh.append(morphed_value)
                month_hour_count += 1
            
            # Calculate and print statistics
            morphed_array = np.array(morphed_rh)
            print("\nMorphing complete!")
            print(f"Morphed RH range: {morphed_array.min():.1f}% to {morphed_array.max():.1f}%")
            print(f"Morphed mean RH: {morphed_array.mean():.1f}%")
            abs_change = morphed_array.mean() - base_data.mean()
            rel_change = (abs_change / base_data.mean() * 100)
            print(f"Average change: {abs_change:+.1f}% ({rel_change:+.1f}%)")
            
            # Sample verification
            print("\nSample verification:")
            print(f"Original first 5: {base_data[:5].tolist()}")
            print(f"Morphed first 5:  {[f'{x:.1f}' for x in morphed_array[:5]]}")
            
            return morphed_rh
            
        except Exception as e:
            print(f"Error in relative humidity morphing: {e}")
            import traceback
            traceback.print_exc()
            return None