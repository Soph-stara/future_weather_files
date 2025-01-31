from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import rasterio

class BaseMorpher:
    def __init__(self, epw_column: int, variable_name: str):
        self.epw_column = epw_column
        self.variable_name = variable_name
        self.epw_path = None
        self.required_files = []

    
    def bilinear_interpolate(self, values: np.ndarray, x_frac: float, y_frac: float) -> float:
        """
        Perform bilinear interpolation between points, handling missing values.
        If some points are missing, use weighted average of available points.
        """
        if values.shape != (2, 2):
            raise ValueError("Values array must be 2x2")
            
        # Create weight matrix
        w00 = (1 - x_frac) * (1 - y_frac)
        w01 = x_frac * (1 - y_frac)
        w10 = (1 - x_frac) * y_frac
        w11 = x_frac * y_frac
        weights = np.array([[w00, w01], [w10, w11]])
        
        # Create mask for valid values
        valid_mask = ~np.isnan(values)
        
        if not np.any(valid_mask):
            return np.nan
        
        # If all values are valid, do normal bilinear interpolation
        if np.all(valid_mask):
            return np.sum(values * weights)
        
        # Otherwise, do weighted average of available points
        valid_weights = weights * valid_mask
        weight_sum = np.sum(valid_weights)
        if weight_sum > 0:
            return np.sum(values * valid_weights) / weight_sum
        
        return np.nan

    def get_value_from_tif(self, tif_path: Path, lat: float, lon: float, scale_factor: float = 1.0) -> Optional[float]:
        """Extract interpolated value from TIF file, handling missing data"""
        try:
            with rasterio.open(tif_path) as src:
                # Get base pixel coordinates
                x_base = (lon - (-180.0)) / src.transform[0]
                y_base = (90.0 - lat) / -src.transform[4]
                
                # Get integer pixel coordinates
                x = int(x_base)
                y = int(y_base)
                
                # Calculate fractional position within cell
                x_frac = x_base - x
                y_frac = y_base - y
                
                # Read 3x3 window to have more points for interpolation
                window = ((y-1, y+2), (x-1, x+2))
                data = src.read(1, window=window)
                
                # Handle no data values
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
                # Try different subwindows if main one has too many NaNs
                for y_offset in range(2):
                    for x_offset in range(2):
                        subdata = data[y_offset:y_offset+2, x_offset:x_offset+2]
                        if np.count_nonzero(~np.isnan(subdata)) >= 2:  # At least 2 valid points
                            value = self.bilinear_interpolate(subdata, x_frac, y_frac)
                            if not np.isnan(value):
                                # Apply scaling
                                scaled_value = value * scale_factor
                                print(f"Read interpolated value {scaled_value:.2f} from {Path(tif_path).name}")
                                print(f"  Used {np.count_nonzero(~np.isnan(subdata))} valid points")
                                return scaled_value
                
                # If we get here, we couldn't get a valid interpolation
                print(f"Warning: Could not get valid interpolated value at {lat:.4f}°N, {lon:.4f}°E")
                print("Available values in 3x3 window:")
                print(data * scale_factor)
                return None
                    
        except Exception as e:
            print(f"Error reading {tif_path}: {e}")
            return None

    def get_monthly_values(self, monthly_files: List[Path], lat: float, lon: float, scale_factor: float = 1.0) -> Optional[List[float]]:
        """Extract monthly values for a specific location from GeoTIFF files using interpolation"""
        monthly_values = []
        
        for file_path in sorted(monthly_files):
            value = self.get_value_from_tif(file_path, lat, lon, scale_factor)
            if value is not None:
                monthly_values.append(float(value))
            else:
                print(f"Warning: Could not read value from {file_path}")
                return None
        
        if len(monthly_values) != 12:
            print(f"Error: Expected 12 monthly values, got {len(monthly_values)}")
            return None
            
        return monthly_values

    def get_location_from_epw(self) -> Optional[Tuple[float, float, str]]:
        """Extract location information from EPW file header"""
        try:
            with open(self.epw_path, 'r') as f:
                location_line = f.readline().strip().split(',')
                city = location_line[1]
                lat = float(location_line[6])
                lon = float(location_line[7])
                print(f"Location from EPW: {lat:.4f}°N, {lon:.4f}°E")
                return lat, lon, city
        except Exception as e:
            print(f"Error reading location from EPW: {e}")
            return None

    def get_coordinates_from_pixels(self, src, x: float, y: float) -> Tuple[float, float]:
        """Convert pixel coordinates back to geographic coordinates"""
        lon = -180.0 + (x * src.transform[0])
        lat = 90.0 + (y * src.transform[4])
        return lon, lat

    def validate_value(self, value: float, variable_type: str, city: str = "") -> bool:
        """Validate values are within reasonable ranges"""
        ranges = {
            "TEMP": {
                "default": (-30, 45),
                "Barcelona": (0, 40),
                "Copenhagen": (-10, 35),
                "Frankfurt": (-10, 35),
                "Geneva": (-10, 35),
                "Hamburg": (-10, 35),
                "London": (-5, 35),
                "Munich": (-15, 35),
                "Oslo": (-20, 35),
                "Stockholm": (-20, 35),
                "Vienna": (-10, 35)
            },
            "WIND": {"default": (0, 30)},
            "RH": {"default": (0, 100)},
            "PREC": {"default": (0, 500)}
        }
        
        if variable_type not in ranges:
            return True
            
        range_dict = ranges[variable_type]
        range_to_use = range_dict.get(city, range_dict["default"])
        return range_to_use[0] <= value <= range_to_use[1]