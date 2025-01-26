# main.py
from pathlib import Path
from utils.epw_morphing_manager import EPWMorphingManager
from utils.epw_to_csv_converter import EPWtoCSVConverter
from morphers.dry_bulb_temp_morpher import TemperatureMorpher
from morphers.relative_humidity_morpher import RelativeHumidityMorpher
from morphers.dew_point_temp_morpher import DewPointMorpher
from morphers.wind_speed_morpher import WindSpeedMorpher
from morphers.prec_water_morpher import PrecipitableWaterMorpher
from visualization.plot_morphed_variables import add_visualization_to_workflow

def get_available_cities():
    return {
        "Barcelona": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Barcelona/ESP_CT_Barcelona.081800_TMYx.epw",
        "Copenhagen": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Copenhagen/DNK_HS_Copenhagen-Kastrup.AP.061800_TMYx.epw",
        "Frankfurt": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Frankfurt/DEU_HE_Offenbach.Wetterpar.106410_TMYx.epw",
        "Geneva": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Geneva/CHE_GE_Geneva.Intl.AP.067000_TMYx.epw",
        "Hamburg": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Hamburg/DEU_HH_Hamburg-Schmidt.AP.101470_TMYx.epw",
        "London": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/London/GBR_ENG_London.Wea.Ctr-St.James.Park.037700_TMYx.epw",
        "Munich": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Munich/DEU_BY_Munich-Theresienwiese.108650_TMYx.epw",
        "Oslo": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Oslo/NOR_OS_Oslo.Blindern.014920_TMYx.epw",
        "Stockholm": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Stockholm/SWE_ST_Stockholm.024850_TMYx.epw",
        "Vienna": "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data/Vienna/AUT_WI_Wien-Innere.Stadt.110340_TMYx.epw"
    }

def main():
    base_path = "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/HadCM3_data"
    output_path = "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/output"
    
    cities = get_available_cities()
    print("\nAvailable cities:", ", ".join(sorted(cities.keys())))
    
    while True:
        city = input("Select a city: ").strip()
        if city in cities:
            epw_path = cities[city]
            break
        print("Invalid city. Please try again.")
    
    manager = EPWMorphingManager(base_path, epw_path, output_path)
    converter = EPWtoCSVConverter(output_path)
    
    print(f"\nProcessing {city}...")
    manager.organize_files()
    
    print("\nAvailable experiments:", ", ".join(manager.experiments))
    while True:
        experiment = input("Select experiment (A2a/A2b/A2c): ").strip()
        if experiment in manager.experiments:
            break
        print("Invalid experiment. Please try again.")

    print("\nAvailable years:", ", ".join(manager.future_years))
    while True:
        target_year = input("Select target year (2020/2050/2080): ").strip()
        if target_year in manager.future_years:
            break
        print("Invalid year. Please try again.")
    
    # Add morphers
    manager.add_morpher(TemperatureMorpher())
    manager.add_morpher(RelativeHumidityMorpher())
    manager.add_morpher(DewPointMorpher())
    manager.add_morpher(WindSpeedMorpher())
    manager.add_morpher(PrecipitableWaterMorpher())

    # Create output directory structure
    output_dir = Path(output_path) / city / f"{experiment}_{target_year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create morphed EPW file
    morphed_epw_path = manager.create_morphed_epw(experiment, target_year, city)
    
    if morphed_epw_path:
        # Convert both original and morphed EPW files to CSV
        original_csv = output_dir / f"{Path(epw_path).stem}.csv"
        morphed_csv = output_dir / f"{Path(morphed_epw_path).stem}.csv"
        
        converter.convert_epw_to_csv(epw_path, str(original_csv))
        converter.convert_epw_to_csv(morphed_epw_path, str(morphed_csv))
        
        # Add visualization
        add_visualization_to_workflow(epw_path, morphed_epw_path, str(output_dir), city, experiment, target_year)

if __name__ == "__main__":
    main()