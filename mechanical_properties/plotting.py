"""
Legacy plotting functions for backwards compatibility.
"""

from .mechanical_property_data import MechanicalPropertyData


def plot_data(file_paths, sampling_freq_hz, plot_title, sensitivity_scaled, excitation_amp):
    """
    Legacy wrapper function for backwards compatibility.
    Processes files using the MechanicalPropertyData class.

    Args:
        file_paths (list): A list of strings, where each string is the full path to a data file.
        sampling_freq_hz (float): The sampling frequency in Hertz.
        plot_title (str): The title for the plot.
        sensitivity_scaled (float): The sensitivity scaling factor.
        excitation_amp (float): The amplitude of the excitation signal.
    """
    for file_path in file_paths:
        try:
            # Create MechanicalPropertyData object
            mech_data = MechanicalPropertyData(
                file_path=file_path,
                sampling_freq_hz=sampling_freq_hz,
                sensitivity_scaled=sensitivity_scaled,
                excitation_amp=excitation_amp
            )

            # Interactive phase selection
            mech_data.select_phase_interactively()

            # Generate excitation signal
            mech_data.generate_excitation_signal()

            # Export data
            mech_data.export_to_csv()

            # Plot all data
            mech_data.plot_all(plot_title=plot_title, save=True)

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            continue
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")
            continue
