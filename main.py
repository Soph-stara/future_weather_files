from pathlib import Path
from utils.epw_morphing_manager import EPWMorphingManager
from utils.epw_to_csv_converter import EPWtoCSVConverter
from morphers.dry_bulb_temp_morpher import TemperatureMorpher
from morphers.relative_humidity_morpher import RelativeHumidityMorpher
from morphers.dew_point_temp_morpher import DewPointMorpher
from morphers.wind_speed_morpher import WindSpeedMorpher
from morphers.prec_water_morpher import PrecipitableWaterMorpher
from morphers.solar_radiation_morpher import SolarRadiationMorpher
from visualization.plot_morphed_variables import add_visualization_to_workflow

def get_available_cities():
    base_path = "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing/data/epw_data"
    return {
        "Barcelona": f"{base_path}/Barcelona/ESP_CT_Barcelona.081800_TMYx.epw",
        "Copenhagen": f"{base_path}/Copenhagen/DNK_HS_Copenhagen-Kastrup.AP.061800_TMYx.epw",
        "Frankfurt": f"{base_path}/Frankfurt/DEU_HE_Offenbach.Wetterpar.106410_TMYx.epw",
        "Geneva": f"{base_path}/Geneva/CHE_GE_Geneva.Intl.AP.067000_TMYx.epw",
        "Hamburg": f"{base_path}/Hamburg/DEU_HH_Hamburg-Schmidt.AP.101470_TMYx.epw",
        "London": f"{base_path}/London/GBR_ENG_London.Wea.Ctr-St.James.Park.037700_TMYx.epw",
        "Munich": f"{base_path}/Munich/DEU_BY_Munich-Theresienwiese.108650_TMYx.epw",
        "Oslo": f"{base_path}/Oslo/NOR_OS_Oslo.Blindern.014920_TMYx.epw",
        "Stockholm": f"{base_path}/Stockholm/SWE_ST_Stockholm.024850_TMYx.epw",
        "Vienna": f"{base_path}/Vienna/AUT_WI_Wien-Innere.Stadt.110340_TMYx.epw"
    }

def get_available_scenarios():
    return ['ssp126', 'ssp245', 'ssp370', 'ssp585']

def get_available_periods():
    return ['2021-2040', '2041-2060', '2061-2080', '2081-2100']

def main():
    base_path = "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing_experiment/data"
    output_path = "/Users/sophiehamann/Documents/Infrared_City/DAP_research_project/New_morphing_tool/epw_morphing_experiment/output"    
    cities = get_available_cities()
    scenarios = get_available_scenarios()
    periods = get_available_periods()
    
    print("\nAvailable cities:", ", ".join(sorted(cities.keys())))
    while True:
        city = input("Select a city: ").strip()
        if city in cities:
            epw_path = cities[city]
            break
        print("Invalid city. Please try again.")
    
    print("\nAvailable scenarios:", ", ".join(scenarios))
    while True:
        scenario = input("Select scenario (e.g., ssp126): ").strip()
        if scenario in scenarios:
            break
        print("Invalid scenario. Please try again.")
        
    print("\nAvailable time periods:", ", ".join(periods))
    while True:
        period = input("Select time period: ").strip()
        if period in periods:
            break
        print("Invalid time period. Please try again.")
    
    manager = EPWMorphingManager(base_path, epw_path, output_path)
    converter = EPWtoCSVConverter(output_path)
    
    print(f"\nProcessing {city}...")
    manager.organize_files(scenario, period)
    
    # Add morphers
    manager.add_morpher(TemperatureMorpher())
    manager.add_morpher(RelativeHumidityMorpher())
    manager.add_morpher(DewPointMorpher())
    manager.add_morpher(WindSpeedMorpher())
    manager.add_morpher(PrecipitableWaterMorpher())
    manager.add_morpher(SolarRadiationMorpher())

    # Create output directory structure
    output_dir = Path(output_path) / city / f"{scenario}_{period}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create morphed EPW file
    morphed_epw_path = manager.create_morphed_epw(scenario, period, city)
    
    if morphed_epw_path:
        # Convert both original and morphed EPW files to CSV
        original_csv = output_dir / f"{Path(epw_path).stem}.csv"
        morphed_csv = output_dir / f"{Path(morphed_epw_path).stem}.csv"
        
        converter.convert_epw_to_csv(epw_path, str(original_csv))
        converter.convert_epw_to_csv(morphed_epw_path, str(morphed_csv))
        
        # Add visualization
        add_visualization_to_workflow(epw_path, morphed_epw_path, str(output_dir), city, scenario, period)

if __name__ == "__main__":
    main()