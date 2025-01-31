import pandas as pd
import os
from pathlib import Path
from typing import Optional

class EPWtoCSVConverter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def get_location_from_epw(self, file_path: str) -> dict:
        """Extract location information from EPW header"""
        with open(file_path, 'r') as epw_file:
            first_line = epw_file.readline()
            metadata = first_line.split(',')

            location = {
                "city": metadata[1].strip(),
                "state": metadata[2].strip(),
                "country": metadata[3].strip(),
                "latitude": float(metadata[6].strip()),
                "longitude": float(metadata[7].strip()),
                "time_zone": int(float(metadata[8].strip())),
                "elevation": float(metadata[9].strip())
            }
            return location

    def convert_epw_to_csv(self, epw_file_path: str, output_path: str) -> str:
        """Convert EPW file to CSV format and save it"""
        location_info = self.get_location_from_epw(epw_file_path)
        print("Location information:")
        print(location_info)

        column_names = [
            "Year", "Month", "Day", "Hour", "Minute", "Data Source and Uncertainty Flags",
            "Dry Bulb Temperature", "Dew Point Temperature", "Relative Humidity", "Atmospheric Station Pressure",
            "Extraterrestrial Horizontal Radiation", "Extraterrestrial Direct Normal Radiation", 
            "Horizontal Infrared Radiation Intensity", "Global Horizontal Radiation", 
            "Direct Normal Radiation", "Diffuse Horizontal Radiation", "Global Horizontal Illuminance", 
            "Direct Normal Illuminance", "Diffuse Horizontal Illuminance", "Zenith Luminance", 
            "Wind Direction", "Wind Speed", "Total Sky Cover", "Opaque Sky Cover", "Visibility", 
            "Ceiling Height", "Present Weather Observation", "Present Weather Codes",
            "Precipitable Water", "Aerosol Optical Depth", "Snow Depth", "Days Since Last Snowfall",
            "Albedo", "Liquid Precipitation Depth", "Liquid Precipitation Quantity"
        ]

        with open(epw_file_path, 'r') as file:
            lines = file.readlines()

        data = []
        for line in lines[8:]:
            data.append(line.strip().split(','))

        df = pd.DataFrame(data, columns=column_names)

        numeric_columns = [
            "Year", "Month", "Day", "Hour", "Minute", "Dry Bulb Temperature", 
            "Dew Point Temperature", "Relative Humidity", "Atmospheric Station Pressure", 
            "Extraterrestrial Horizontal Radiation", "Extraterrestrial Direct Normal Radiation", 
            "Horizontal Infrared Radiation Intensity", "Global Horizontal Radiation", 
            "Direct Normal Radiation", "Diffuse Horizontal Radiation",
            "Global Horizontal Illuminance", "Direct Normal Illuminance", 
            "Diffuse Horizontal Illuminance", "Zenith Luminance", "Wind Direction", 
            "Wind Speed", "Total Sky Cover", "Opaque Sky Cover", "Visibility", 
            "Ceiling Height", "Precipitable Water", "Aerosol Optical Depth", 
            "Snow Depth", "Days Since Last Snowfall", "Albedo", 
            "Liquid Precipitation Depth", "Liquid Precipitation Quantity"
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nData saved to: {output_path}")
        
        return str(output_path)