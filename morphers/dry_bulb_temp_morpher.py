from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class TemperatureMorpher(BaseMorpher):
    def __init__(self):
        super().__init__(epw_column=6, variable_name='TEMP')
        self.required_files = ['TEMP_historic', 'TMAX_historic', 'TMIN_historic', 
                             'TMAX_future', 'TMIN_future']
    
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

    def transfer_function(self, x: float, m: float = 1.0, n: float = 1.0) -> float:
        """Calculate transfer function g = x^m * (1-x)^n"""
        return np.power(x, m) * np.power(1 - x, n)
    
    def normalize_temperature(self, T: float, T_min: float, T_max: float) -> float:
        """Normalize temperature to [0,1] range"""
        return (T - T_min) / (T_max - T_min) if T_max != T_min else 0.5
    
    def calculate_btws_parameters(self, T: float, T_mean: float, T_min: float, T_max: float,
                              delta_mean: float, delta_max: float, delta_min: float) -> Tuple[float, float, float]:
        """Calculate parameters for BTWS morphing following Eames et al."""
        # Normalize temperature to [0,1]
        x = (T - T_min) / (T_max - T_min) if T_max != T_min else 0.5
        
        # Calculate future temperature bounds
        T_min_future = T_min + delta_min
        T_max_future = T_max + delta_max
        
        # Calculate mean-preserving scaling factor (equation 16)
        T_mean_future = T_mean + delta_mean
        if T_max_future != T_min_future and T_mean != T_min:
            S = ((T_mean_future - T_min_future) / (T_max_future - T_min_future) * 
                 (T_max - T_min) / (T_mean - T_min)) - 1
        else:
            S = 0
            
        return x, S, T_min_future, T_max_future
    
    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                      scenario: str, period: str) -> Optional[List[float]]:
        try:
            print("\nProcessing temperature morphing using Eames BTWS method...")
            
            # Calculate monthly statistics for base temperature
            monthly_stats = self.calculate_monthly_statistics(base_data)
            print(f"Original temperature range: {base_data.min():.1f}°C to {base_data.max():.1f}°C")
            
            # Get all required data as before
            location_info = self.get_location_from_epw()
            if not location_info:
                return None
            lat, lon, city = location_info
            
            # Process historic and future data
            hist_temp_monthly = self.get_monthly_values(climate_files['TEMP_historic'], lat, lon, scale_factor=0.1)
            hist_tmax_monthly = self.get_monthly_values(climate_files['TMAX_historic'], lat, lon, scale_factor=0.1)
            hist_tmin_monthly = self.get_monthly_values(climate_files['TMIN_historic'], lat, lon, scale_factor=0.1)
            
            if not all([hist_temp_monthly, hist_tmax_monthly, hist_tmin_monthly]):
                return None
            
            future_tmax = self.get_value_from_tif(climate_files['TMAX_future'][0], lat, lon, scale_factor=0.1)
            future_tmin = self.get_value_from_tif(climate_files['TMIN_future'][0], lat, lon, scale_factor=0.1)
            
            if future_tmax is None or future_tmin is None:
                return None
            
            future_temp = (future_tmax + future_tmin) / 2
            
            # Apply BTWS morphing
            morphed_temps = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, T in enumerate(base_data):
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Get current month statistics
                T_mean = monthly_stats[current_month]['mean']
                T_min = monthly_stats[current_month]['min']
                T_max = monthly_stats[current_month]['max']
                
                # Calculate deltas for current month
                delta_mean = future_temp - np.mean(hist_temp_monthly)
                delta_max = future_tmax - hist_tmax_monthly[current_month - 1]
                delta_min = future_tmin - hist_tmin_monthly[current_month - 1]
                
                # Get BTWS parameters
                x, S, T_min_future, T_max_future = self.calculate_btws_parameters(
                    T, T_mean, T_min, T_max, delta_mean, delta_max, delta_min
                )
                
                # Calculate transfer function (using m=n=1 as per paper's simplest version)
                g = x * (1 - x)
                
                # Calculate g_mean for normalization using daily values
                day_start = (hour // 24) * 24
                day_temps = base_data[day_start:day_start + 24]
                g_values = [(t - T_min) / (T_max - T_min) * (1 - (t - T_min) / (T_max - T_min)) 
                           for t in day_temps if T_max != T_min]
                g_mean = np.mean(g_values) if g_values else 0
                
                # Apply BTWS transformation
                if g_mean != 0:
                    x_prime = x + (S * g) / g_mean
                else:
                    # If g_mean is 0, fall back to simple shift to preserve mean change
                    x_prime = x
                
                # Ensure bounds are preserved
                x_prime = max(0, min(1, x_prime))
                
                # Convert back to temperature
                morphed_temp = T_min_future + x_prime * (T_max_future - T_min_future)
                
                # Validate transformation preserved mean change
                if not (T_min_future <= morphed_temp <= T_max_future):
                    # Fall back to simple shift if bounds are violated
                    morphed_temp = T + delta_mean
                
                # Validate and append
                if not self.validate_value(morphed_temp, 'TEMP', city):
                    print(f"Warning: Morphed temperature {morphed_temp:.2f}°C at hour {hour} outside expected range for {city}")
                
                morphed_temps.append(morphed_temp)
                month_hour_count += 1
            
            # Print verification stats
            morphed_array = np.array(morphed_temps)
            print("\nMorphing complete!")
            print(f"Original temperature range: {base_data.min():.1f}°C to {base_data.max():.1f}°C")
            print(f"Morphed temperature range: {morphed_array.min():.1f}°C to {morphed_array.max():.1f}°C")
            print(f"Mean temperature change: {(morphed_array.mean() - base_data.mean()):.1f}°C")
            
            return morphed_temps
            
        except Exception as e:
            print(f"Error in temperature morphing: {e}")
            import traceback
            traceback.print_exc()
            return None