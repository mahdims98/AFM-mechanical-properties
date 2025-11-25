import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
import sys


def parse_voltage(value_str):
    """
    Parses a string with a voltage value and unit (mV, uV, V)
    and returns the value in millivolts (mV).
    """
    value_str = str(value_str).strip()
    try:
        if 'mV' in value_str:
            return float(value_str.replace('mV', ''))
        elif 'uV' in value_str:
            return float(value_str.replace('uV', '')) / 1000.0
        elif 'V' in value_str:
            return float(value_str.replace('V', '')) * 1000.0
        else:
            return float(value_str)
    except (ValueError, TypeError):
        return None


class MechanicalPropertyData:
    """
    A class to store and process mechanical property measurement data.

    Attributes:
        file_path (str): Path to the data file
        sampling_freq_hz (float): Sampling frequency in Hz
        sensitivity_scaled (float): Sensitivity scaling factor (nm/mV)
        excitation_amp (float): Amplitude of excitation signal in meters
        data (pd.DataFrame): Processed data with Time, Deflection, and Excitation
        phase_shift_deg (float): Phase shift in degrees
        excitation_freq_hz (float): Excitation frequency in Hz
        base_name (str): Base name of the file without extension
    """

    def __init__(self, file_path, sampling_freq_hz, sensitivity_scaled, excitation_amp):
        """
        Initialize the MechanicalPropertyData object.

        Args:
            file_path (str): Path to the data file
            sampling_freq_hz (float): Sampling frequency in Hz
            sensitivity_scaled (float): Sensitivity scaling factor
            excitation_amp (float): Amplitude of excitation signal
        """
        self.file_path = file_path
        self.sampling_freq_hz = sampling_freq_hz
        self.sensitivity_scaled = sensitivity_scaled
        self.excitation_amp = excitation_amp
        self.sampling_period_s = 1.0 / sampling_freq_hz

        self.data = None
        self.phase_shift_deg = 0
        self.excitation_freq_hz = None
        self.base_name = os.path.basename(file_path)
        self.file_name_no_ext = re.sub(r'\.(txt|csv)$', '', self.base_name)
        self.base_dir = os.path.dirname(file_path)

        # Load and process the data
        self._load_data()

    def _load_data(self):
        """Load and preprocess the raw data from file."""
        # Read the data using the first row as header, only reading the second column (Deflection)
        data = pd.read_csv(self.file_path, sep=';', header=0, usecols=[1])
        data.columns = ['Deflection']

        # Convert 'Deflection' column to numeric values in nm
        data['Deflection_mV'] = data['Deflection'].apply(parse_voltage) * self.sensitivity_scaled

        # Drop rows where parsing failed
        data.dropna(subset=['Deflection_mV'], inplace=True)

        # Reset index after dropping NaN values
        data.reset_index(drop=True, inplace=True)

        # Generate sample numbers based on row index
        data['Index'] = data.index

        # Convert sample number to time in microseconds
        data['Time_us'] = data['Index'] * self.sampling_period_s * 1e6

        self.data = data
        self.excitation_freq_hz = self.sampling_freq_hz / len(data)

    def select_phase_interactively(self):
        """
        Display a plot for the user to select the time instant for excitation maximum.
        Calculates the phase shift based on the selected point.
        """
        print("Please click on the deflection plot to select the time instant for excitation maximum.")
        print("Close the plot window after selecting the point.")

        # Create figure for initial selection
        fig_select, ax_select = plt.subplots(figsize=(12, 4))
        ax_select.plot(self.data['Time_us'], self.data['Deflection_mV'],
                      label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax_select.set_title(f'Deflection vs Time (Click to select time instant)')
        ax_select.set_xlabel('Time (µs)')
        ax_select.set_ylabel('Deflection (nm)')
        ax_select.grid(True)
        ax_select.legend()

        # Store selected point
        selected_time = {'time_us': None}

        def onclick(event):
            if event.inaxes == ax_select and event.xdata is not None:
                selected_time['time_us'] = event.xdata
                # Remove previous vertical line if exists
                for line in ax_select.lines[1:]:
                    line.remove()
                # Draw vertical line at selected point
                ax_select.axvline(x=event.xdata, color='red', linestyle='--',
                                linewidth=2, label=f'Selected: {event.xdata:.2f} µs')
                ax_select.legend()
                fig_select.canvas.draw()
                print(f"Selected time: {event.xdata:.2f} µs")

        # Connect click event
        cid = fig_select.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

        # Calculate phase shift based on selection
        if selected_time['time_us'] is None:
            print("No point selected. Using default phase shift of 0°.")
            self.phase_shift_deg = 0
        else:
            selected_time_s = selected_time['time_us'] / 1e6
            phase_shift_rad = (np.pi / 2) - (2 * np.pi * self.excitation_freq_hz * selected_time_s)
            self.phase_shift_deg = np.degrees(phase_shift_rad)
            # Normalize to [-180, 180] range
            self.phase_shift_deg = ((self.phase_shift_deg + 180) % 360) - 180
            print(f"Calculated phase shift: {self.phase_shift_deg:.2f}°")

    def set_phase_shift(self, phase_shift_deg):
        """
        Manually set the phase shift in degrees.

        Args:
            phase_shift_deg (float): Phase shift in degrees
        """
        self.phase_shift_deg = phase_shift_deg
        print(f"Phase shift set to: {self.phase_shift_deg:.2f}°")

    def generate_excitation_signal(self):
        """Generate the excitation signal with the current phase shift."""
        # Generate time array in seconds
        time_s = self.data['Index'] * self.sampling_period_s

        # Convert phase shift to radians
        phase_shift_rad = np.radians(self.phase_shift_deg)

        # Generate sinusoidal excitation signal with phase shift
        self.data['Excitation'] = self.excitation_amp * np.sin(
            2 * np.pi * self.excitation_freq_hz * time_s + phase_shift_rad
        )

    def export_to_csv(self, output_path=None):
        """
        Export the processed data to CSV.

        Args:
            output_path (str, optional): Custom output path. If None, saves to same directory as input file.

        Returns:
            str: Path to the exported file
        """
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        # Create export dataframe
        export_data = pd.DataFrame({
            'Index': self.data['Index'],
            'Deflection_nm': self.data['Deflection_mV'],
            'Excitation': self.data['Excitation']
        })

        # Determine output path
        if output_path is None:
            export_filename = f"{self.file_name_no_ext}_processed.csv"
            output_path = os.path.join(self.base_dir, export_filename)

        # Export to CSV
        export_data.to_csv(output_path, index=False)
        print(f"Processed data exported to '{output_path}'")
        return output_path

    def plot_all(self, plot_title=None, save=True):
        """
        Create and display all three plots: Deflection vs Time, Excitation vs Time,
        and Deflection vs Excitation.

        Args:
            plot_title (str, optional): Title for the plots. If None, uses filename.
            save (bool): Whether to save the plot to file
        """
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        if plot_title is None:
            plot_title = self.file_name_no_ext

        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Plot 1: Deflection vs Time
        ax1.plot(self.data['Time_us'], self.data['Deflection_mV'],
                label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax1.set_title(f'{plot_title} - Deflection vs Time')
        ax1.set_xlabel('Time (µs)')
        ax1.set_ylabel('Deflection (nm)')
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Excitation vs Time
        ax2.plot(self.data['Time_us'], self.data['Excitation'], linewidth=1.1, color='red')
        ax2.set_title(f'{plot_title} - Excitation vs Time (f = {self.excitation_freq_hz:.2f} Hz, φ = {self.phase_shift_deg:.2f}°)')
        ax2.set_xlabel('Time (µs)')
        ax2.set_ylabel('Excitation (nm)')
        ax2.grid(True)

        # Plot 3: Deflection vs Excitation
        ax3.plot(self.data['Excitation'], self.data['Deflection_mV'], linewidth=1.1, color='green')
        ax3.set_title(f'{plot_title} - Deflection vs Excitation')
        ax3.set_xlabel('Excitation (nm)')
        ax3.set_ylabel('Deflection (nm)')
        ax3.grid(True)

        plt.tight_layout()

        if save:
            # Create the 'plots' subfolder if it doesn't exist
            plots_dir = os.path.join(self.base_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            # Create a filesystem-safe filename
            save_filename = f"{plot_title.replace(' ', '_').replace('.', '')}_all_plots.png"
            full_save_path = os.path.join(plots_dir, save_filename)

            plt.savefig(full_save_path, dpi=300)
            print(f"Plots saved to '{full_save_path}'")

        plt.show()

    def plot_deflection(self, save=False):
        """
        Plot only the deflection vs time.

        Args:
            save (bool): Whether to save the plot to file
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.data['Time_us'], self.data['Deflection_mV'],
               label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax.set_title('Deflection vs Time')
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Deflection (nm)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        if save:
            plots_dir = os.path.join(self.base_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f"{self.file_name_no_ext}_deflection.png")
            plt.savefig(save_path, dpi=300)
            print(f"Deflection plot saved to '{save_path}'")

        plt.show()

    def plot_hysteresis(self, save=False):
        """
        Plot the hysteresis curve (Deflection vs Excitation).

        Args:
            save (bool): Whether to save the plot to file
        """
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.data['Excitation'], self.data['Deflection_mV'], linewidth=1.1, color='green')
        ax.set_title('Deflection vs Excitation (Hysteresis)')
        ax.set_xlabel('Excitation (nm)')
        ax.set_ylabel('Deflection (nm)')
        ax.grid(True)
        plt.tight_layout()

        if save:
            plots_dir = os.path.join(self.base_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f"{self.file_name_no_ext}_hysteresis.png")
            plt.savefig(save_path, dpi=300)
            print(f"Hysteresis plot saved to '{save_path}'")

        plt.show()

    def get_data(self):
        """
        Get the processed data as a pandas DataFrame.

        Returns:
            pd.DataFrame: Processed data
        """
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()
        return self.data.copy()

    def get_statistics(self):
        """
        Calculate and return statistical properties of the data.

        Returns:
            dict: Dictionary with statistical measures
        """
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        stats = {
            'deflection_mean': self.data['Deflection_mV'].mean(),
            'deflection_std': self.data['Deflection_mV'].std(),
            'deflection_min': self.data['Deflection_mV'].min(),
            'deflection_max': self.data['Deflection_mV'].max(),
            'deflection_range': self.data['Deflection_mV'].max() - self.data['Deflection_mV'].min(),
            'excitation_freq_hz': self.excitation_freq_hz,
            'phase_shift_deg': self.phase_shift_deg,
            'num_samples': len(self.data),
            'duration_us': self.data['Time_us'].max()
        }
        return stats

    def select_indices_for_slope(self):
        """
        Display the deflection vs time plot for the user to interactively select
        two points by clicking. Returns the indices of the selected points.

        Returns:
            tuple: (index1, index2) - Indices of the two selected points (ordered)
        """
        print("Please click on TWO points on the deflection-time plot to select the range for slope calculation.")
        print("Close the plot window after selecting both points.")

        # Create figure for selection
        fig_select, ax_select = plt.subplots(figsize=(12, 4))
        ax_select.plot(self.data['Time_us'], self.data['Deflection_mV'],
                      label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax_select.set_title('Deflection vs Time (Click to select TWO points for slope calculation)')
        ax_select.set_xlabel('Time (µs)')
        ax_select.set_ylabel('Deflection (nm)')
        ax_select.grid(True)
        ax_select.legend()

        # Store selected points
        selected_points = []
        vertical_lines = []

        def onclick(event):
            if event.inaxes == ax_select and event.xdata is not None and len(selected_points) < 2:
                # Find the nearest index
                time_us = event.xdata
                idx = (self.data['Time_us'] - time_us).abs().idxmin()
                selected_points.append(idx)

                # Draw vertical line at selected point
                color = 'red' if len(selected_points) == 1 else 'orange'
                vline = ax_select.axvline(x=self.data.loc[idx, 'Time_us'], color=color,
                                         linestyle='--', linewidth=2,
                                         label=f'Point {len(selected_points)}: idx={idx}')
                vertical_lines.append(vline)
                ax_select.legend()
                fig_select.canvas.draw()
                print(f"Point {len(selected_points)} selected: Index={idx}, Time={self.data.loc[idx, 'Time_us']:.2f} µs")

                if len(selected_points) == 2:
                    print("Two points selected. You can close the window now.")

        # Connect click event
        cid = fig_select.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

        # Validate selection
        if len(selected_points) < 2:
            print("Warning: Less than 2 points selected. Cannot calculate slope.")
            return None, None

        # Return indices in order (smaller first)
        idx1, idx2 = sorted(selected_points)
        print(f"\nSelected range: Index {idx1} to {idx2} ({idx2 - idx1 + 1} points)")
        return idx1, idx2

    def calculate_slope(self, idx1=None, idx2=None, interactive=True):
        """
        Calculate the slope of the deflection vs excitation plot between two indices
        using least squares linear regression.

        Args:
            idx1 (int, optional): Starting index. If None and interactive=True,
                                 will prompt user to select interactively.
            idx2 (int, optional): Ending index. If None and interactive=True,
                                 will prompt user to select interactively.
            interactive (bool): If True and indices not provided, prompts interactive selection.

        Returns:
            dict: Dictionary containing slope, intercept, R-squared, and other fit parameters
        """
        # Ensure excitation signal is generated
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        # Get indices interactively if not provided
        if idx1 is None or idx2 is None:
            if interactive:
                idx1, idx2 = self.select_indices_for_slope()
                if idx1 is None or idx2 is None:
                    print("Error: Invalid indices selected.")
                    return None
            else:
                print("Error: Indices must be provided when interactive=False")
                return None

        # Validate indices
        if idx1 >= idx2:
            print("Error: idx1 must be less than idx2")
            return None
        if idx1 < 0 or idx2 >= len(self.data):
            print(f"Error: Indices out of range. Valid range: 0 to {len(self.data)-1}")
            return None

        # Extract data for the selected range
        excitation_range = self.data.loc[idx1:idx2, 'Excitation'].values
        deflection_range = self.data.loc[idx1:idx2, 'Deflection_mV'].values

        # Perform least squares linear fit: deflection = slope * excitation + intercept
        # Using numpy's polyfit for linear regression (degree 1)
        coefficients = np.polyfit(excitation_range, deflection_range, 1)
        slope = coefficients[0]
        intercept = coefficients[1]

        # Calculate fitted values and residuals
        deflection_fit = slope * excitation_range + intercept
        residuals = deflection_range - deflection_fit
        ss_res = np.sum(residuals**2)  # Residual sum of squares
        ss_tot = np.sum((deflection_range - np.mean(deflection_range))**2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate standard error of the slope
        n = len(excitation_range)
        if n > 2:
            std_error = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((excitation_range - np.mean(excitation_range))**2))
        else:
            std_error = np.nan

        # Prepare results
        results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'std_error': std_error,
            'idx1': idx1,
            'idx2': idx2,
            'num_points': n,
            'excitation_range': excitation_range,
            'deflection_range': deflection_range,
            'deflection_fit': deflection_fit
        }

        # Print results
        print("\n" + "="*60)
        print("SLOPE CALCULATION RESULTS")
        print("="*60)
        print(f"Index range: {idx1} to {idx2} ({n} points)")
        print(f"Slope: {slope:.6e} nm/nm (dimensionless)")
        print(f"Intercept: {intercept:.6f} nm")
        print(f"R-squared: {r_squared:.6f}")
        print(f"Standard error: {std_error:.6e}")
        print("="*60 + "\n")

        return results

    def plot_slope_fit(self, slope_results):
        """
        Visualize the slope fit on the deflection vs excitation plot.

        Args:
            slope_results (dict): Results dictionary from calculate_slope()
        """
        if slope_results is None:
            print("Error: No slope results to plot.")
            return

        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Full deflection vs excitation with highlighted region
        ax1.plot(self.data['Excitation'], self.data['Deflection_mV'],
                linewidth=1.1, color='lightgray', alpha=0.5, label='Full data')
        ax1.plot(slope_results['excitation_range'], slope_results['deflection_range'],
                linewidth=2, color='blue', marker='o', markersize=3, label='Selected range')
        ax1.set_title('Deflection vs Excitation (Full View)')
        ax1.set_xlabel('Excitation (nm)')
        ax1.set_ylabel('Deflection (nm)')
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Zoomed view with linear fit
        ax2.scatter(slope_results['excitation_range'], slope_results['deflection_range'],
                   color='blue', alpha=0.6, s=20, label='Data points')
        ax2.plot(slope_results['excitation_range'], slope_results['deflection_fit'],
                color='red', linewidth=2, linestyle='--', label='Linear fit')
        ax2.set_title(f'Linear Fit (Slope = {slope_results["slope"]:.4f}, R² = {slope_results["r_squared"]:.4f})')
        ax2.set_xlabel('Excitation (nm)')
        ax2.set_ylabel('Deflection (nm)')
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(self.base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"{self.file_name_no_ext}_slope_fit.png")
        plt.savefig(save_path, dpi=300)
        print(f"Slope fit plot saved to '{save_path}'")

        plt.show()


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

if __name__ == '__main__':
    # --- Configuration ---
    # Directory where the files are located. Use a raw string (r"...") for Windows paths.
    directory = r"D:\PhD local\LBNI\Mechanical Property Measurments\Code - calculation of mechanical properties\Marcos Data"
    # --- Directory Validation ---
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        print("Please check the path and try again.")
        sys.exit(1)  # Exit the script if the directory is not found

    filename = "example_4_circle.csv"
    file_path = os.path.join(directory, filename)

    excitation_amp = 100e-9
    sampling_frequency = 1e6  # 1 MHz
    sensitivity = 1  # nm/V
    sensitivity_scaled = sensitivity / 10**3

    # --- NEW CLASS-BASED APPROACH ---
    # Create a MechanicalPropertyData object
    mech_data = MechanicalPropertyData(
        file_path=file_path,
        sampling_freq_hz=sampling_frequency,
        sensitivity_scaled=sensitivity_scaled,
        excitation_amp=excitation_amp
    )

    # Interactively select phase shift
    mech_data.select_phase_interactively()

    # Generate excitation signal
    mech_data.generate_excitation_signal()

    # Export processed data
    mech_data.export_to_csv()

    # Plot all data
    mech_data.plot_all(plot_title=filename, save=True)

    # --- EXAMPLE: Access data for other calculations ---
    # Get the processed data as a DataFrame
    data_df = mech_data.get_data()
    print("\nData shape:", data_df.shape)
    print("\nFirst few rows:")
    print(data_df.head())

    # Get statistics
    stats = mech_data.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # --- EXAMPLE: Calculate slope of deflection vs excitation ---
    # Method 1: Interactive selection - user clicks on deflection-time plot
    print("\n" + "="*60)
    print("SLOPE CALCULATION - Interactive Mode")
    print("="*60)
    slope_results = mech_data.calculate_slope(interactive=True)

    # Visualize the slope fit
    if slope_results is not None:
        mech_data.plot_slope_fit(slope_results)

    # Method 2: Manual specification of indices (if you know the range)
    # Example: Calculate slope between indices 100 and 200
    # slope_results = mech_data.calculate_slope(idx1=100, idx2=200, interactive=False)
    # if slope_results is not None:
    #     mech_data.plot_slope_fit(slope_results)

    # --- OPTIONAL: Plot individual plots ---
    # mech_data.plot_deflection(save=False)
    # mech_data.plot_hysteresis(save=False)

    # --- OLD APPROACH (still works for backwards compatibility) ---
    # plot_data([file_path], sampling_frequency, filename, sensitivity_scaled, excitation_amp)