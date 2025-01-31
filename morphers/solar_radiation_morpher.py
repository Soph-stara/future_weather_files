from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from morphers.base_morpher import BaseMorpher

class SolarRadiationMorpher(BaseMorpher):
    def __init__(self):
        super().__init__(epw_column=13, variable_name='SOLRAD')
        # Define required files for solar radiation morphing
        self.required_files = ['SRAD_historic']
        
    def transfer_function(self, x: float, m: float = 1.0, n: float = 1.0) -> float:
        """Calculate transfer function g = x^m * (1-x)^n"""
        return np.power(x, m) * np.power(1 - x, n)
    
    def normalize_radiation(self, rad: float, rad_min: float, rad_max: float) -> float:
        """Normalize radiation to [0,1] range"""
        return (rad - rad_min) / (rad_max - rad_min) if rad_max != rad_min else 0.5

    def get_monthly_solar_radiation(self, historic_dir: Path, lat: float, lon: float) -> Optional[List[float]]:
        """Get monthly solar radiation values from historic data"""
        monthly_values = []
        srad_dir = historic_dir / 'wc2.1_2.5m_srad'
        
        for month in range(1, 13):
            month_str = str(month).zfill(2)
            file_name = f'wc2.1_2.5m_srad_{month_str}.tif'
            file_path = srad_dir / file_name
            
            print(f"Reading solar radiation from: {file_path}")
            value = self.get_value_from_tif(file_path, lat, lon)
            
            if value is None:
                print(f"Warning: Could not read solar radiation for month {month}")
                return None
                
            monthly_values.append(value)
        
        return monthly_values

    def morph_variable(self, base_data: pd.Series, climate_files: Dict[str, Path],
                      scenario: str, period: str) -> Optional[List[float]]:
        try:
            print("\nProcessing solar radiation morphing using BWS method...")
            
            # Calculate statistics for base radiation data
            rad_min = base_data.min()
            rad_max = base_data.max()
            rad_mean = base_data.mean()
            
            print(f"Original radiation range: {rad_min:.1f} to {rad_max:.1f} W/m²")
            
            # Get location info
            location_info = self.get_location_from_epw()
            if not location_info:
                return None
            lat, lon, city = location_info
            print(f"Processing location: {city} ({lat:.4f}°N, {lon:.4f}°E)")
            
            # Get historic directory path
            base_dir = climate_files['TMAX_future'][0].parent.parent.parent
            historic_dir = base_dir.parent / 'historic'
            print(f"Reading historic data from: {historic_dir}")
            
            # Get historic monthly solar radiation values
            historic_rad = self.get_monthly_solar_radiation(historic_dir, lat, lon)
            if historic_rad is None:
                return None
                
            print(f"Historic monthly solar radiation values loaded: {len(historic_rad)} months")
            
            # Get temperature change
            future_tmax = self.get_value_from_tif(climate_files['TMAX_future'][0], lat, lon, scale_factor=0.1)
            future_tmin = self.get_value_from_tif(climate_files['TMIN_future'][0], lat, lon, scale_factor=0.1)
            if future_tmin is None or future_tmax is None:
                print("Could not read future temperature data")
                return None
                
            # Define hours per month for analysis
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            # Calculate change factors for solar radiation
            # Use historic records to establish seasonal patterns
            monthly_means = []
            month_hour_count = 0
            current_month = 1
            month_data = []
            
            for rad in base_data:
                month_data.append(rad)
                month_hour_count += 1
                if month_hour_count >= hours_per_month[current_month - 1]:
                    monthly_means.append(np.mean([x for x in month_data if x > 0]))  # Only consider daylight hours
                    month_data = []
                    month_hour_count = 0
                    current_month += 1
            
            # Calculate seasonal scaling factors
            historic_monthly_rad = np.array(historic_rad)
            future_monthly_rad = historic_monthly_rad.copy()
            
            # Estimate future radiation changes by month
            # Use a more conservative approach with smaller changes
            winter_months = [0, 1, 11]  # Dec, Jan, Feb
            summer_months = [5, 6, 7]   # Jun, Jul, Aug
            
            for i in range(12):
                if i in winter_months:
                    # Smaller changes in winter when radiation is already low
                    change_factor = -0.05  # 5% reduction
                elif i in summer_months:
                    # Moderate changes in summer
                    change_factor = -0.08  # 8% reduction
                else:
                    # Spring/autumn changes
                    change_factor = -0.06  # 6% reduction
                    
                future_monthly_rad[i] *= (1 + change_factor)
            
            # Calculate overall scaling factor based on monthly changes
            monthly_changes = (future_monthly_rad - historic_monthly_rad) / historic_monthly_rad
            S = np.clip(np.mean(monthly_changes), -0.10, 0.10)  # Limit to ±10%
            
            print(f"Monthly radiation changes: {[f'{x*100:.1f}%' for x in monthly_changes]}")
            print(f"Winter mean change: {np.mean(monthly_changes[winter_months])*100:.1f}%")
            print(f"Summer mean change: {np.mean(monthly_changes[summer_months])*100:.1f}%")
            print(f"Final scaling factor S: {S:.3f}")
            # Print average monthly change
            print(f"Average monthly change: {np.mean(monthly_changes)*100:.1f}%")
            print(f"Calculated scaling factor S: {S:.4f}")
            
            # Apply BWS morphing
            morphed_rads = []
            current_month = 1
            month_hour_count = 0
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            
            for hour, rad in enumerate(base_data):
                if month_hour_count >= hours_per_month[current_month - 1]:
                    month_hour_count = 0
                    current_month += 1
                    if current_month > 12:
                        current_month = 1
                
                # Normalize to [0,1]
                x = self.normalize_radiation(rad, rad_min, rad_max)
                
                # Calculate transfer function
                g = self.transfer_function(x)
                
                # Calculate g_mean for the day to normalize
                day_start = (hour // 24) * 24
                day_rads = base_data[day_start:day_start + 24]
                g_values = [self.transfer_function(self.normalize_radiation(r, rad_min, rad_max)) 
                           for r in day_rads]
                g_mean = np.mean(g_values) if g_values else g
                
                # Apply BWS transformation
                if g_mean != 0:
                    x_prime = x + (S * g) / g_mean
                else:
                    x_prime = x
                
                # Ensure bounds are preserved
                x_prime = max(0, min(1, x_prime))
                
                # Convert back to radiation
                morphed_rad = rad_min + x_prime * (rad_max - rad_min)
                
                # Validate and append
                if not self.validate_value(morphed_rad, 'SOLRAD', city):
                    print(f"Warning: Morphed radiation {morphed_rad:.2f} W/m² outside expected range for {city}")
                    
                morphed_rads.append(morphed_rad)
                month_hour_count += 1
            
            # Print verification stats
            morphed_array = np.array(morphed_rads)
            print("\nMorphing complete!")
            print(f"Original radiation range: {rad_min:.1f} to {rad_max:.1f} W/m²")
            print(f"Morphed radiation range: {morphed_array.min():.1f} to {morphed_array.max():.1f} W/m²")
            print(f"Mean radiation change: {(morphed_array.mean() - rad_mean):.1f} W/m²")
            
            return morphed_rads
            
        except Exception as e:
            print(f"Error in solar radiation morphing: {e}")
            import traceback
            traceback.print_exc()
            return None