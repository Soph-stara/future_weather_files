import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

class EPWVisualizer:
    def __init__(self, output_dir: Path):
        # Create plots directory within the output directory
        self.output_dir = output_dir / 'plots'
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Visualization output directory: {self.output_dir}")
        plt.style.use('default')

    def create_comparison_plot(self, original_data: pd.Series, morphed_data: pd.Series, 
                             variable_name: str, units: str, original_units: str = None,
                             city: str = "", scenario: str = "", period: str = ""):
        try:
            plt.figure(figsize=(15, 8))
            
            hours = range(len(original_data))
            
            # Plot the data
            plt.plot(hours, original_data, label='Original', color='#1f77b4', 
                    linewidth=1, alpha=0.8)
            plt.plot(hours, morphed_data, label='Morphed', color='#ff7f0e', 
                    linewidth=1, alpha=0.8)
            
            # Calculate statistics
            original_mean = original_data.mean()
            morphed_mean = morphed_data.mean()
            absolute_change = morphed_mean - original_mean
            relative_change = (absolute_change / original_mean * 100) if original_mean != 0 else 0
            
            # Create statistics text
            stats_text = (
                f'Original Mean: {original_mean:.2f} {units}\n'
                f'Morphed Mean: {morphed_mean:.2f} {units}\n'
                f'Absolute Change: {absolute_change:+.2f} {units}\n'
                f'Relative Change: {relative_change:+.1f}%'
            )
            
            # Set title and labels
            title = f'{variable_name} Comparison - {city}'
            if scenario and period:
                title += f'\n{scenario.upper()} {period}'
            plt.title(title, pad=20)
            plt.xlabel('Hours of Year')
            plt.ylabel(f'{variable_name} ({units})')
            
            # Adjust y-axis limits to add padding
            y_min = min(original_data.min(), morphed_data.min())
            y_max = max(original_data.max(), morphed_data.max())
            y_range = y_max - y_min
            padding = 0.05 * y_range  # 5% padding
            plt.ylim(y_min - padding, y_max + padding)
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add stats text box
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='white', alpha=0.8))
            
            # Create month borders
            hours_per_month = [31*24, 28*24, 31*24, 30*24, 31*24, 30*24, 
                             31*24, 31*24, 30*24, 31*24, 30*24, 31*24]
            cumulative_hours = np.cumsum(hours_per_month)
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for i, hour in enumerate(cumulative_hours[:-1]):
                plt.axvline(x=hour, color='gray', linestyle='--', alpha=0.3)
                plt.text(hour - hours_per_month[i]/2, y_min - padding*2, 
                        month_names[i], ha='center')
            
            # Save plot
            plot_filename = f'{variable_name.lower().replace(" ", "_")}_comparison.png'
            plot_path = self.output_dir / plot_filename
            print(f"Saving plot to: {plot_path}")
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            if plot_path.exists():
                print(f"Successfully created plot: {plot_path}")
            else:
                print(f"Warning: Plot file was not created at {plot_path}")
                
        except Exception as e:
            print(f"Error creating plot for {variable_name}: {str(e)}")
            plt.close()
            raise

def create_visualizations(original_csv: str, morphed_csv: str, output_dir: str, 
                        city: str, scenario: str, period: str):
    try:
        print("\nReading CSV files...")
        original_data = pd.read_csv(original_csv)
        morphed_data = pd.read_csv(morphed_csv)
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        visualizer = EPWVisualizer(output_path)
        
        # Define variables to plot with their units
        variables_to_plot = [
            ('Dry Bulb Temperature', 'Dry Bulb Temperature', '°C', None),
            ('Relative Humidity', 'Relative Humidity', '%', None),
            ('Wind Speed', 'Wind Speed', 'm/s', 'knots'),
            ('Precipitable Water', 'Precipitable Water', 'mm', None),
            ('Dew Point Temperature', 'Dew Point Temperature', '°C', None),
            ('Global Horizontal Radiation', 'Solar Radiation', 'W/m²', None)
        ]
        
        for col, name, unit, orig_unit in variables_to_plot:
            if col in original_data.columns and col in morphed_data.columns:
                print(f"\nProcessing {name}...")
                visualizer.create_comparison_plot(
                    original_data[col],
                    morphed_data[col],
                    name,
                    unit,
                    orig_unit,
                    city,
                    scenario,
                    period
                )
            else:
                print(f"Warning: Column {col} not found in data files")
                
        # Create composite plot
        create_composite_plot(original_data, morphed_data, output_path, city, scenario, period)
                
    except Exception as e:
        print(f"Error in visualization creation: {str(e)}")
        raise

def create_composite_plot(original_data: pd.DataFrame, morphed_data: pd.DataFrame, 
                        output_dir: Path, city: str, scenario: str, period: str):
    """Create a composite plot showing all variables"""
    try:
        plt.figure(figsize=(20, 12))
        
        variables = [
            ('Dry Bulb Temperature', '°C'),
            ('Relative Humidity', '%'),
            ('Wind Speed', 'm/s'),
            ('Precipitable Water', 'mm'),
            ('Dew Point Temperature', '°C'),
            ('Global Horizontal Radiation', 'W/m²')
        ]
        
        n_vars = len(variables)
        
        for i, (var, unit) in enumerate(variables, 1):
            if var in original_data.columns and var in morphed_data.columns:
                plt.subplot(n_vars, 1, i)
                
                hours = range(len(original_data))
                plt.plot(hours, original_data[var], label='Original', 
                        color='#1f77b4', linewidth=1, alpha=0.8)
                plt.plot(hours, morphed_data[var], label='Morphed', 
                        color='#ff7f0e', linewidth=1, alpha=0.8)
                
                plt.ylabel(f'{var}\n({unit})')
                if i == 1:
                    title = f'Climate Variable Comparisons - {city}'
                    if scenario and period:
                        title += f'\n{scenario.upper()} {period}'
                    plt.title(title)
                if i == n_vars:
                    plt.xlabel('Hours of Year')
                
                plt.grid(True, alpha=0.3)
                plt.legend()
        
        plt.tight_layout()
        composite_path = output_dir / 'plots' / 'composite_comparison.png'
        plt.savefig(composite_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nCreated composite plot: {composite_path}")
        
    except Exception as e:
        print(f"Error creating composite plot: {str(e)}")
        plt.close()
        raise

def add_visualization_to_workflow(epw_path: str, morphed_epw_path: str, 
                                output_dir: str, city: str, scenario: str, period: str):
    """Add visualization step to the main EPW morphing workflow"""
    try:
        original_csv = str(Path(output_dir) / f"{Path(epw_path).stem}.csv")
        morphed_csv = str(Path(output_dir) / f"{Path(morphed_epw_path).stem}.csv")
        
        print(f"\nStarting visualization workflow for {city} ({scenario} {period})...")
        print(f"EPW path: {epw_path}")
        print(f"Morphed EPW path: {morphed_epw_path}")
        print(f"Output directory: {output_dir}")
        print(f"Looking for original CSV: {original_csv}")
        print(f"Looking for morphed CSV: {morphed_csv}")
        
        create_visualizations(original_csv, morphed_csv, output_dir, city, scenario, period)
        print(f"Visualization complete for {city} ({scenario} {period})!")
        
    except Exception as e:
        print(f"Error in visualization process: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 7:
        print("Usage: python plot_morphed_variables.py original.csv morphed.csv output_dir city scenario period")
        sys.exit(1)
    
    create_visualizations(sys.argv[1], sys.argv[2], sys.argv[3], 
                         sys.argv[4], sys.argv[5], sys.argv[6])