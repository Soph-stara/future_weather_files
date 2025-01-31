from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class WindSpeedMorpher(BaseMorpher):
    """Implementation of wind speed morphing algorithm using GeoTIFF data"""
    
    def __init__(self):
        # Wind speed is in column 21 of EPW file
        super().__init__(epw_column=21, variable_name='WIND')
        # Update required files for new data structure
        self.required_files = ['WIND_historic']
    
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

    def convert_knots_to_ms(self, speed_knots: float) -> float:
        """Convert wind speed from knots to m/s"""
        return speed_knots * 0.514444

    def convert_ms_to_knots(self, speed_ms: float) -> float:
        """Convert wind speed from m/s to knots"""
        return speed_ms / 0.514444

    def normalize_wind_speeds(self, hist_wind_monthly: List[float], base_monthly_stats: Dict) -> Tuple[List[float], List[float]]:
        """
        Normalize both historic and base wind speeds
        Returns normalized historic and base values for each month
        """
        # Calculate means
        hist_mean = np.mean(hist_wind_monthly)
        base_values = [stats['mean'] for stats in base_monthly_stats.values()]
        base_mean = np.mean(base_values)
        
        # Normalize
        hist_norm = [h/hist_mean for h in hist_wind_monthly]
        base_norm = [b/base_mean for b in base_values]
        
        return hist_norm, base_norm

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                      scenario: str, period: str) -> Optional[List[float]]:
        try:
            print("\nProcessing wind speed morphing...")
            
            # Get location using base class method
            location_info = self.get_location_from_epw()
            if not location_info:
                return None
            lat, lon, city = location_info
            
            # Get historic monthly wind speeds
            print("\nReading historic wind speed data...")
            hist_wind = self.get_monthly_values(climate_files['WIND_historic'], lat, lon)
            if not hist_wind:
                return None
            
            # Calculate base statistics
            base_monthly_stats = self.calculate_monthly_statistics(base_data)
            
            # Print original statistics
            print("\nWind Speed morphing statistics:")
            print(f"Original wind speed range: {base_data.min():.1f} to {base_data.max():.1f} knots")
            print(f"Original mean wind speed: {base_data.mean():.1f} knots")
            
            # Normalize data
            print("\nCalculating monthly scaling factors...")
            print("Debug: Monthly wind speeds and changes")
            hist_norm, base_norm = self.normalize_wind_speeds(hist_wind, base_monthly_stats)
            
            # Calculate scaling factors
            scaling_factors = []
            for month in range(12):
                # Calculate WIND_m using normalized values
                wind_change = ((hist_norm[month] - base_norm[month]) / base_norm[month] * 100) if base_norm[month] > 0 else 0
                
                # Apply formula αws_m = 1 + WIND_m/100
                alpha = 1 + (wind_change / 100)
                scaling_factors.append(alpha)
                
                print(f"Month {month + 1}:")
                print(f"  Base mean (original): {self.convert_knots_to_ms(base_monthly_stats[month + 1]['mean']):.3f} m/s")
                print(f"  Historical mean (original): {hist_wind[month]:.3f} m/s")
                print(f"  Base mean (normalized): {base_norm[month]:.3f}")
                print(f"  Historical mean (normalized): {hist_norm[month]:.3f}")
                print(f"  WIND_m: {wind_change:.1f}%")
                print(f"  α = {alpha:.3f}")
            
            # Apply morphing equation
            print("\nApplying morphing equation...")
            morphed_speeds = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, ws0 in enumerate(base_data):
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Apply formula: ws = (αws_m · ws0) · 0.514444
                alpha = scaling_factors[current_month - 1]
                morphed_speed = (alpha * ws0) * 0.514444
                
                morphed_speeds.append(morphed_speed)
                month_hour_count += 1
            
            # Calculate verification statistics
            morphed_array = np.array(morphed_speeds)
            base_data_ms = base_data.apply(self.convert_knots_to_ms)
            
            print("\nVerification statistics:")
            print(f"Original range: {base_data.min():.1f} to {base_data.max():.1f} knots")
            print(f"Morphed range: {self.convert_ms_to_knots(morphed_array.min()):.1f} to "
                  f"{self.convert_ms_to_knots(morphed_array.max()):.1f} knots")
            
            abs_change = morphed_array.mean() - base_data_ms.mean()
            rel_change = (abs_change / base_data_ms.mean() * 100) if base_data_ms.mean() != 0 else 0
            print(f"Mean change: {abs_change:.2f} m/s ({rel_change:+.1f}%)")
            
            print("\nSample verification (first 5 values):")
            print(f"Original (knots): {base_data[:5].tolist()}")
            print(f"Morphed (m/s): {[f'{x:.1f}' for x in morphed_array[:5]]}")
            
            return morphed_speeds
            
        except Exception as e:
            print(f"Error in wind speed morphing: {e}")
            import traceback
            traceback.print_exc()
            return None