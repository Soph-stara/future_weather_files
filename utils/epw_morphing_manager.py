from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from morphers.base_morpher import BaseMorpher
from morphers.dew_point_temp_morpher import DewPointMorpher
from morphers.dry_bulb_temp_morpher import TemperatureMorpher
from morphers.prec_water_morpher import PrecipitableWaterMorpher
from morphers.relative_humidity_morpher import RelativeHumidityMorpher
from morphers.wind_speed_morpher import WindSpeedMorpher
from morphers.solar_radiation_morpher import SolarRadiationMorpher
import shutil

class EPWMorphingManager:
    """Manages the morphing of multiple EPW variables"""
    
    def __init__(self, base_path: str, epw_path: str, output_dir: str):
        """Initialize the EPW Morphing Manager"""
        self.base_path = Path(base_path)
        self.epw_path = Path(epw_path)
        self.output_dir = Path(output_dir)
        self.morphers: List[BaseMorpher] = []
        self.organized_files: Dict = {}
        self.historic_path = self.base_path / "historic"
        self.future_path = self.base_path / "future"
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
    
    def add_morpher(self, morpher: BaseMorpher) -> None:
        """Add a variable morpher to the manager"""
        self.morphers.append(morpher)
    
    # In EPWMorphingManager class
    def organize_files(self, scenario: str, period: str) -> None:
        """Organize files by scenario and period"""
        self.organized_files = {}
        
        # Update patterns to use 30s instead of 5m
        monthly_patterns = {
            'TEMP': {'dir': 'wc2.1_2.5m_tavg', 'pattern': 'wc2.1_2.5m_tavg_{:02d}.tif'},
            'TMAX': {'dir': 'wc2.1_2.5m_tmax', 'pattern': 'wc2.1_2.5m_tmax_{:02d}.tif'},
            'TMIN': {'dir': 'wc2.1_2.5m_tmin', 'pattern': 'wc2.1_2.5m_tmin_{:02d}.tif'},
            'PREC': {'dir': 'wc2.1_2.5m_prec', 'pattern': 'wc2.1_2.5m_prec_{:02d}.tif'},
            'WIND': {'dir': 'wc2.1_2.5m_wind', 'pattern': 'wc2.1_2.5m_wind_{:02d}.tif'},
            'VAPR': {'dir': 'wc2.1_2.5m_vapr', 'pattern': 'wc2.1_2.5m_vapr_{:02d}.tif'},
            'SRAD': {'dir': 'wc2.1_2.5m_srad', 'pattern': 'wc2.1_2.5m_srad_{:02d}.tif'},
            'hist_bio': {'dir': 'wc2.1_2.5m_bio', 'pattern': 'wc2.1_2.5m_bio_{}.tif'}  # Note: removed :02d as files use single digits
        }
        
        print("\nOrganizing climate data files...")
            
        # Organize historic files
        print("Reading historic files from:", self.historic_path)
        for var, info in monthly_patterns.items():
            monthly_files = []
            var_dir = self.historic_path / info['dir']
            
            if not var_dir.exists():
                print(f"Warning: Directory not found: {var_dir}")
                continue
                
            for month in range(1, 13):
                file_path = var_dir / info['pattern'].format(month)
                if file_path.exists():
                    monthly_files.append(file_path)
            
            if len(monthly_files) == 12:
                print(f"Found {len(monthly_files)} historic files for {var}")
                self.organized_files[f"{var}_historic"] = sorted(monthly_files)
            else:
                print(f"Warning: Missing historic files for {var}, found {len(monthly_files)} files")
        
        
        
        
        # Organize future files
        future_scenario_path = self.future_path / scenario / period
        print(f"\nReading future files from: {future_scenario_path}")

        future_vars = ['TMAX', 'TMIN', 'PREC']
        for var in future_vars:
            var_lower = var.lower()
            file_name = f"wc2.1_2.5m_{var_lower}_UKESM1-0-LL_{scenario}_{period}.tif"
            file_path = future_scenario_path / file_name
            
            if file_path.exists():
                print(f"Found future file for {var}: {file_path}")
                self.organized_files[f"{var}_future"] = [file_path]
            else:
                print(f"Warning: Future file not found: {file_path}")
        
        # Add future bioclim file
        file_name = f"wc2.1_2.5m_bio_UKESM1-0-LL_{scenario}_{period}.tif"
        file_path = future_scenario_path / file_name
        if file_path.exists():
            print(f"Found future bioclim file: {file_path}")
            self.organized_files['future_bio'] = [file_path]
        else:
            print(f"Warning: Future bioclim file not found: {file_path}")
    
    def read_epw_data(self, column: int) -> Optional[pd.Series]:
        """Read specific column from EPW file"""
        try:
            epw_data = pd.read_csv(self.epw_path, skiprows=8, header=None)
            return epw_data.iloc[:, column]
        except Exception as e:
            print(f"Error reading EPW file: {e}")
            return None
    
    def create_morphed_epw(self, scenario: str, period: str, city: str) -> Optional[str]:
        """Create new EPW file with all morphed variables"""
        try:
            # Create output directory structure
            output_subdir = self.output_dir / city / f"{scenario}_{period}"
            output_subdir.mkdir(parents=True, exist_ok=True)

            original_name = self.epw_path.stem
            output_filename = f"{original_name}_morphed_{scenario}_{period}.epw"
            output_path = output_subdir / output_filename

            # Copy original EPW to the output directory
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
                    
                    # Pass the epw_path to the morpher
                    morpher.epw_path = self.epw_path
                    
                    base_data = self.read_epw_data(morpher.epw_column)
                    if base_data is None:
                        print(f"Error: Could not read base data for {morpher.variable_name}")
                        continue
                        
                    # Check required files
                    if not all(req in self.organized_files for req in morpher.required_files):
                        missing = [req for req in morpher.required_files if req not in self.organized_files]
                        print(f"Error: Missing required files for {morpher.variable_name}: {missing}")
                        print(f"Available files: {list(self.organized_files.keys())}")
                        continue
                    
                    # Special handling for DewPointMorpher
                    if isinstance(morpher, DewPointMorpher):
                        if 'TEMP' not in stored_results or 'RH' not in stored_results:
                            print("Error: Temperature and RH must be morphed before dew point")
                            continue
                        
                        morphed_values = morpher.morph_variable(
                            base_data, 
                            self.organized_files,
                            scenario, 
                            period,
                            future_temp=pd.Series(stored_results['TEMP']),
                            future_rh=pd.Series(stored_results['RH'])
                        )
                    else:
                        morphed_values = morpher.morph_variable(
                            base_data, 
                            self.organized_files,
                            scenario, 
                            period
                        )
                    
                    if morphed_values:
                        # Store morphed values both for EPW file and for potential dependencies
                        morphed_data[morpher.epw_column] = morphed_values
                        stored_results[morpher.variable_name] = morphed_values
                    else:
                        print(f"Warning: Morphing failed for {morpher.variable_name}")
                
                if not morphed_data:
                    print("Error: No variables were successfully morphed")
                    return None
                    
                # Create new EPW data lines
                new_data_lines = []
                for hour, line in enumerate(data_lines):
                    elements = line.strip().split(',')
                    
                    for column, values in morphed_data.items():
                        elements[column] = f"{values[hour]:.1f}"
                    
                    new_line = ','.join(elements) + '\n'
                    new_data_lines.append(new_line)
                
                # Write new EPW file
                with open(output_path, 'w') as f:
                    f.writelines(header_lines)
                    f.writelines(new_data_lines)
                
                print(f"\nCreated morphed EPW file: {output_path}")
                return str(output_path)
            
            except Exception as e:
                print(f"Error processing EPW file: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        except Exception as e:
            print(f"Error creating morphed EPW file: {e}")
            import traceback
            traceback.print_exc()
            return None