# utils/epw_morphing_manager.py

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from morphers.base_morpher import BaseMorpher
from morphers.dew_point_temp_morpher import DewPointMorpher

__all__ = ['EPWMorphingManager']

class EPWMorphingManager:
    """Manages the morphing of multiple EPW variables"""
    
    def __init__(self, base_path: str, epw_path: str, output_dir: str):
        """Initialize the EPW Morphing Manager"""
        self.base_path = Path(base_path)
        self.epw_path = Path(epw_path)
        self.output_dir = Path(output_dir)
        self.morphers: List[BaseMorpher] = []
        self.organized_files: Dict = {}
        self.experiments = ['A2a', 'A2b', 'A2c']
        self.future_years = ['2020', '2050', '2080']
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
            
    def add_morpher(self, morpher: BaseMorpher) -> None:
        """Add a variable morpher to the manager"""
        self.morphers.append(morpher)
        
    def organize_files(self) -> None:
        """Organize files by experiment, year, and variable"""
        self.organized_files = {exp: {year: {} for year in self.future_years + ['1980']} 
                              for exp in self.experiments}
        
        # Find all files
        for file_path in self.base_path.glob("*.*"):
            if file_path.suffix not in ['.dif', '.mea']:
                continue
                
            # Parse filename
            parts = file_path.stem.split('_')
            if len(parts) != 4:  # HADCM3_A2a_TEMP_2020
                continue
                
            experiment = parts[1]  # A2a, A2b, A2c
            variable = parts[2]    # TEMP, TMAX, etc.
            year = parts[3]        # 2020, 2050, 2080, 1980
            
            # Store file path in organized structure
            if variable not in self.organized_files[experiment][year]:
                self.organized_files[experiment][year][variable] = []
            self.organized_files[experiment][year][variable].append(file_path)
            
    def read_epw_data(self, column: int) -> Optional[pd.Series]:
        """Read specific column from EPW file"""
        try:
            epw_data = pd.read_csv(self.epw_path, skiprows=8, header=None)
            return epw_data.iloc[:, column]
        except Exception as e:
            print(f"Error reading EPW file: {e}")
            return None
            
    def create_morphed_epw(self, experiment: str, target_year: str, city: str) -> Optional[str]:
        """Create new EPW file with all morphed variables"""
        # Create city/experiment/year specific output directory
        output_subdir = self.output_dir / city / f"{experiment}_{target_year}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        original_name = self.epw_path.stem
        output_filename = f"{original_name}_morphed_{experiment}_{target_year}.epw"
        output_path = output_subdir / output_filename

        # Copy original EPW to the output directory
        import shutil
        original_epw_dest = output_subdir / self.epw_path.name
        shutil.copy2(self.epw_path, original_epw_dest)
        
        try:
            # Read original EPW file
            with open(self.epw_path, 'r') as f:
                header_lines = [next(f) for _ in range(8)]
                data_lines = f.readlines()
                
            # Process each variable
            morphed_data = {}
            stored_results = {}  # Store intermediate results for dependent morphers
            
            for morpher in self.morphers:
                print(f"\nProcessing {morpher.variable_name}...")
                print(f"EPW column index: {morpher.epw_column}")
                
                base_data = self.read_epw_data(morpher.epw_column)
                if base_data is None:
                    print(f"Error: Could not read base data for {morpher.variable_name}")
                    continue
                    
                climate_files = self.organized_files[experiment][target_year]
                
                # Special handling for DewPointMorpher
                if isinstance(morpher, DewPointMorpher):
                    if 'TEMP' not in stored_results or 'RH' not in stored_results:
                        print("Error: Temperature and RH must be morphed before dew point")
                        continue
                    
                    morphed_values = morpher.morph_variable(
                        base_data, 
                        climate_files,
                        experiment, 
                        target_year,
                        future_temp=pd.Series(stored_results['TEMP']),
                        future_rh=pd.Series(stored_results['RH'])
                    )
                else:
                    # Check required climate files
                    if not all(key in climate_files for key in morpher.required_files):
                        print(f"Error: Missing required climate files for {morpher.variable_name}")
                        print(f"Required files: {morpher.required_files}")
                        print(f"Available files: {list(climate_files.keys())}")
                        continue
                    
                    morphed_values = morpher.morph_variable(
                        base_data, 
                        climate_files, 
                        experiment, 
                        target_year
                    )
                
                if morphed_values:
                    # Debug output before storing
                    print("\nSample data verification:")
                    print(f"Original first 5: {base_data[:5].tolist()}")
                    print(f"Morphed first 5:  {morphed_values[:5]}")
                    print(f"Original mean: {base_data.mean():.2f}")
                    print(f"Morphed mean:  {sum(morphed_values)/len(morphed_values):.2f}")
                    print(f"Change: {((sum(morphed_values)/len(morphed_values) - base_data.mean()) / base_data.mean() * 100):+.1f}%")
                    
                    # Store morphed values both for EPW file and for potential dependencies
                    morphed_data[morpher.epw_column] = morphed_values
                    stored_results[morpher.variable_name] = morphed_values
                else:
                    print(f"Warning: Morphing failed for {morpher.variable_name}")
            
            if not morphed_data:
                print("Error: No variables were successfully morphed")
                return None
                
            # Debug: Print which columns will be modified
            print("\nColumns to be modified:")
            for col in morphed_data.keys():
                print(f"Column {col}")
                
            # Create new EPW file with morphed values
            new_data_lines = []
            for hour, line in enumerate(data_lines):
                elements = line.strip().split(',')
                
                # Debug first line before modification
                if hour == 0:
                    print(f"\nFirst line before modification: {','.join(elements)}")
                
                # Update each morphed variable
                for column, values in morphed_data.items():
                    old_value = elements[column]
                    elements[column] = f"{values[hour]:.1f}"
                    
                    # Debug first few changes
                    if hour < 5:
                        print(f"Hour {hour}, Column {column}: {old_value} -> {elements[column]}")
                
                # Create new line and add to list
                new_line = ','.join(elements) + '\n'
                new_data_lines.append(new_line)
                
                # Debug first line after modification
                if hour == 0:
                    print(f"First line after modification: {new_line.strip()}")
            
            # Write new EPW file
            with open(output_path, 'w') as f:
                f.writelines(header_lines)
                f.writelines(new_data_lines)
                
            # Verify the output file
            print(f"\nVerifying output file {output_path}...")
            verification_data = pd.read_csv(output_path, skiprows=8, header=None)
            for column in morphed_data.keys():
                original_mean = self.read_epw_data(column).mean()
                morphed_mean = verification_data.iloc[:, column].mean()
                print(f"Column {column} verification:")
                print(f"Original mean: {original_mean:.2f}")
                print(f"Morphed mean: {morphed_mean:.2f}")
                print(f"Change: {((morphed_mean - original_mean) / original_mean * 100):+.1f}%")
            
            print(f"\nCreated modified EPW file: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error creating morphed EPW file: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def convert_epw_to_csv(self, epw_path: str, output_path: str) -> Optional[str]:
        """Convert EPW file to CSV format"""
        try:
            # Read EPW file
            epw_data = pd.read_csv(epw_path, skiprows=8, header=None)
            
            # Define column names based on EPW file structure
            columns = [
                'Year', 'Month', 'Day', 'Hour', 'Minute',
                'Data Source and Uncertainty Flags',
                'Dry Bulb Temperature', 'Dew Point Temperature',
                'Relative Humidity', 'Atmospheric Station Pressure',
                'Extraterrestrial Horizontal Radiation',
                'Extraterrestrial Direct Normal Radiation',
                'Horizontal Infrared Radiation Intensity',
                'Global Horizontal Radiation',
                'Direct Normal Radiation', 'Diffuse Horizontal Radiation',
                'Global Horizontal Illuminance',
                'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance',
                'Zenith Luminance', 'Wind Direction', 'Wind Speed',
                'Total Sky Cover', 'Opaque Sky Cover', 'Visibility',
                'Ceiling Height', 'Present Weather Observation',
                'Present Weather Codes', 'Precipitable Water',
                'Aerosol Optical Depth', 'Snow Depth',
                'Days Since Last Snowfall', 'Albedo',
                'Liquid Precipitation Depth', 'Liquid Precipitation Quantity'
            ]
            
            # Assign column names
            epw_data.columns = columns
            
            # Save to CSV
            epw_data.to_csv(output_path, index=False)
            print(f"Converted EPW file saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Error converting EPW to CSV: {e}")
            return None