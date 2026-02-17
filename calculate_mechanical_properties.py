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
    directory = r"D:\PhD local\LBNI\Mechanical Property Measurments\Code - Anlysis of Nanoscope data"

    # --- Directory Validation ---
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        print("Please check the path and try again.")
        sys.exit(1)

    # --- File and Parameters ---
    filename = "cycle_100_data.csv"
    file_path = os.path.join(directory, filename)

    excitation_amp = 150          # Excitation amplitude in nanometers (100 nm)
    sampling_frequency = 6.25e6         # Sampling frequency in Hz (1 MHz)
    sensitivity = 35.17                 # Sensitivity in nm/V
    tip_radius = 2                   # Tip radius in nm for Young's Modulus calculation
    sensitivity_scaled = sensitivity # Convert to nm/mV -> not required for nanoscope
    spring_constant = 0.5           # Spring constant in N/m (or nN/nm)
    deformation_percentage = 15     # Percentage of max excitation for deformation calc
    sample_poisson_ratio = 0.3      # Poisson's ratio of the sample
    cantilever_poisson_ratio = 0.17 # Poisson's ratio of the cantilever tip
    cantilever_E_GPa = 169.0        # Young's Modulus of the cantilever tip in GPa
    neglect_cantilever_effect = True # Whether to neglect the cantilever effect in Modulus calc

    # --- Create MechanicalPropertyData Object ---
    mech_data = MechanicalPropertyData(
        file_path=file_path,
        sampling_freq_hz=sampling_frequency,
        sensitivity_scaled=sensitivity_scaled,
        excitation_amp=excitation_amp,
        spring_constant_N_m=spring_constant
    )

    # --- Interactively Select Phase Shift ---
    mech_data.select_phase_interactively()

    # --- Remove DC Offset (Optional) ---
    mech_data.remove_dc_offset_interactively()

    # --- Set Processing Mode (Optional) ---
    mech_data.set_processing_mode(use_force=True)

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
    # Method 1: Interactive selection - user clicks on deflection-time plot to select range for slope on Deflection vs Indentation
    print("\n" + "="*60)
    print("SLOPE CALCULATION - Interactive Mode")
    print("="*60)

    #**********************************
    slope_results = mech_data.calculate_slope_with_intercept(interactive=True)

    # Visualize the slope fit
    if slope_results is not None:
        mech_data.plot_slope_fit(slope_results)

    # deformation
    deformation_results = mech_data.calculate_deformation(percentage=deformation_percentage)
    mech_data.plot_deformation(deformation_results)


    # Dissipation
    area_results = mech_data.calculate_hysteresis_area()
    mech_data.plot_hysteresis_area()

    # Adhesion
    adhesion_results = mech_data.calculate_adhesion()
    mech_data.plot_adhesion(adhesion_results)

    # --- Calculate DMT Modulus ---
    dmt_results = mech_data.calculate_young_modulus_dmt(
        high_factor_pct=90, 
        low_range_pct=30, 
        tip_radius_nm=tip_radius,
        sample_poisson_ratio=sample_poisson_ratio,
        cantilever_poisson_ratio=cantilever_poisson_ratio,
        cantilever_E_GPa=cantilever_E_GPa,
        neglect_cantilever_effect=neglect_cantilever_effect
    )
    mech_data.plot_dmt_fit(dmt_results)


    # --- Calculate Reduced Young's Modulus ---
    if slope_results and deformation_results:
        er_results = mech_data.calculate_reduced_young_modulus(slope_results, deformation_results, tip_radius)

    #print(er_results)

    # Method 2: Manual specification of indices (if you know the range)
    # Example: Calculate slope between indices 100 and 200

    #slope_results = mech_data.calculate_slope_with_intercept(idx1=280, idx2=310, interactive=False)
    #if slope_results is not None:
    #     mech_data.plot_slope_fit(slope_results)

    # --- Optional: Plot Individual Plots ---
    # mech_data.plot_deflection(save=False)
    # mech_data.plot_hysteresis(save=False)
