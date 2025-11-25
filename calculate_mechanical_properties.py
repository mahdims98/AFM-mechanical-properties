"""
Main script for calculating mechanical properties from deflection measurements.

This script demonstrates how to use the mechanical_properties package to:
1. Load and process measurement data
2. Interactively select phase shift
3. Generate excitation signals
4. Calculate slopes using least squares fitting
5. Export and visualize results
"""

import os
import sys

from mechanical_properties import MechanicalPropertyData


if __name__ == '__main__':
    # --- Configuration ---
    # Directory where the files are located. Use a raw string (r"...") for Windows paths.
    directory = r"D:\PhD local\LBNI\Mechanical Property Measurments\Code - calculation of mechanical properties\Marcos Data"

    # --- Directory Validation ---
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        print("Please check the path and try again.")
        sys.exit(1)

    # --- File and Parameters ---
    filename = "example_5_PS.csv"
    file_path = os.path.join(directory, filename)

    excitation_amp = 100          # Excitation amplitude in meters (100 nm)
    sampling_frequency = 1e6         # Sampling frequency in Hz (1 MHz)
    sensitivity = 1                  # Sensitivity in nm/V
    sensitivity_scaled = sensitivity / 10**3  # Convert to nm/mV

    # --- Create MechanicalPropertyData Object ---
    mech_data = MechanicalPropertyData(
        file_path=file_path,
        sampling_freq_hz=sampling_frequency,
        sensitivity_scaled=sensitivity_scaled,
        excitation_amp=excitation_amp
    )

    # --- Interactively Select Phase Shift ---
    mech_data.select_phase_interactively()

    # --- Generate Excitation Signal ---
    mech_data.generate_excitation_signal()

    # --- Export Processed Data ---
    mech_data.export_to_csv()

    # --- Plot All Data ---
    mech_data.plot_all(plot_title=filename, save=True)

    # --- Access Data for Analysis ---
    data_df = mech_data.get_data()
    print("\nData shape:", data_df.shape)
    print("\nFirst few rows:")
    print(data_df.head())

    # --- Get Statistics ---
    stats = mech_data.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # --- Calculate Slope of Deflection vs Excitation ---
    # Method 1: Interactive selection - user clicks on deflection-time plot
    print("\n" + "="*60)
    print("SLOPE CALCULATION - Interactive Mode")
    print("="*60)

    #**********************************
    '''slope_results = mech_data.calculate_slope_with_intercept(interactive=True)

    # Visualize the slope fit
    if slope_results is not None:
        mech_data.plot_slope_fit(slope_results)

    # deformation
    deformation_results = mech_data.calculate_deformation()
    mech_data.plot_deformation(deformation_results)

    # 

    '''
    # Dissipation
    area_results = mech_data.calculate_hysteresis_area()
    mech_data.plot_hysteresis_area()
    # Method 2: Manual specification of indices (if you know the range)
    # Example: Calculate slope between indices 100 and 200
    slope_results = mech_data.calculate_slope_with_intercept(idx1=280, idx2=310, interactive=False)
    if slope_results is not None:
         mech_data.plot_slope_fit(slope_results)

    # --- Optional: Plot Individual Plots ---
    # mech_data.plot_deflection(save=False)
    # mech_data.plot_hysteresis(save=False)
