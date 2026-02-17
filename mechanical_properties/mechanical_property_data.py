"""
Main class for processing mechanical property measurement data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import re

from .utils import parse_voltage


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
        spring_constant_N_m (float): Spring constant in N/m (or nN/nm)
        use_force (bool): Flag to determine if calculations use Force (True) or Deflection (False)
    """

    def __init__(self, file_path, sampling_freq_hz, sensitivity_scaled, excitation_amp, spring_constant_N_m=None):
        """
        Initialize the MechanicalPropertyData object.

        Args:
            file_path (str): Path to the data file
            sampling_freq_hz (float): Sampling frequency in Hz
            sensitivity_scaled (float): Sensitivity scaling factor
            excitation_amp (float): Amplitude of excitation signal
            spring_constant_N_m (float, optional): Spring constant in N/m. Defaults to None.
        """
        self.file_path = file_path
        self.sampling_freq_hz = sampling_freq_hz
        self.sensitivity_scaled = sensitivity_scaled
        self.excitation_amp = excitation_amp
        self.sampling_period_s = 1.0 / sampling_freq_hz
        self.spring_constant_N_m = spring_constant_N_m
        self.use_force = False  # Default to using Deflection

        self.data = None
        self.phase_shift_deg = 0
        self.excitation_freq_hz = None
        self.base_name = os.path.basename(file_path)
        self.file_name_no_ext = re.sub(r'\.(txt|csv)$', '', self.base_name)
        self.base_dir = os.path.dirname(file_path)
        self.max_deflection_idx = None  # Index of maximum deflection point selected by user
        self.contact_point_idx = None   # Index of contact point selected by user
        self.deformation = None  # Calculated deformation value

        # Load and process the data
        self._load_data()
        
        # Calculate force if spring constant is provided
        if self.spring_constant_N_m is not None:
            self.calculate_force()

    def calculate_force(self):
        """Calculate Force (nN) from Deflection (nm) and Spring Constant (N/m)."""
        if self.spring_constant_N_m is not None and self.data is not None:
            # 1 N/m = 1 nN/nm. So Deflection(nm) * k(N/m) = Force(nN)
            self.data['Force_nN'] = self.data['Deflection_nm'] * self.spring_constant_N_m
            print("Force calculated (nN).")

    def set_processing_mode(self, use_force=False):
        """Set whether to use Force or Deflection for calculations."""
        if use_force:
            if self.spring_constant_N_m is None:
                print("Error: Spring constant not set. Cannot use Force mode.")
                return
            if 'Force_nN' not in self.data.columns:
                self.calculate_force()
        self.use_force = use_force
        print(f"Processing mode set to: {'Force' if use_force else 'Deflection'}")

    def _get_signal_info(self):
        """Returns column name, data series, and unit label based on current mode."""
        if self.use_force and 'Force_nN' in self.data.columns:
            return 'Force_nN', self.data['Force_nN'], 'Force (nN)'
        return 'Deflection_nm', self.data['Deflection_nm'], 'Deflection (nm)'

    def _load_data(self):
        """Load and preprocess the raw data from file."""
        # Read the data using the first row as header, only reading the second column (Deflection)
        data = pd.read_csv(self.file_path, sep=';', header=0, usecols=[1])
        data.columns = ['Deflection']

        # Convert 'Deflection' column to numeric values in nm
        data['Deflection_nm'] = data['Deflection'].apply(parse_voltage) * self.sensitivity_scaled

        # Drop rows where parsing failed
        data.dropna(subset=['Deflection_nm'], inplace=True)

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
        Display a plot for the user to select the time instant for excitation maximum
        and the contact point.
        Calculates the phase shift based on the selected peak point.
        """
        print("Please click on the deflection plot to select TWO points:")
        print("1. The time instant for excitation maximum (Peak) - Red dashed line")
        print("2. The contact point - Green dash-dot line")
        print("Clicking a third time will reset the selections.")
        print("Close the plot window after selecting both points.")

        # Create figure for initial selection
        fig_select, ax_select = plt.subplots(figsize=(12, 4))
        ax_select.plot(self.data['Time_us'], self.data['Deflection_nm'],
                      label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax_select.set_title(f'Deflection vs Time (Select Peak then Contact Point)')
        ax_select.set_xlabel('Time (µs)')
        ax_select.set_ylabel('Deflection (nm)')
        ax_select.grid(True)
        ax_select.legend()

        # Store selected points
        selections = {'peak': None, 'contact': None}
        lines = {'peak': None, 'contact': None}

        def onclick(event):
            if event.inaxes == ax_select and event.xdata is not None:
                if selections['peak'] is None:
                    # Select Peak
                    selections['peak'] = event.xdata
                    lines['peak'] = ax_select.axvline(x=event.xdata, color='red', linestyle='--',
                                                    linewidth=2, label='Peak')
                    print(f"Peak selected: {event.xdata:.2f} µs. Now select Contact Point.")
                elif selections['contact'] is None:
                    # Select Contact
                    selections['contact'] = event.xdata
                    lines['contact'] = ax_select.axvline(x=event.xdata, color='green', linestyle='-.',
                                                       linewidth=2, label='Contact Point')
                    print(f"Contact Point selected: {event.xdata:.2f} µs. Selections complete.")
                else:
                    # Reset and start with Peak
                    print("Resetting selections. Peak updated.")
                    selections['peak'] = event.xdata
                    selections['contact'] = None
                    
                    if lines['peak']: lines['peak'].remove()
                    if lines['contact']: lines['contact'].remove()
                    
                    lines['peak'] = ax_select.axvline(x=event.xdata, color='red', linestyle='--',
                                                    linewidth=2, label='Peak')
                    lines['contact'] = None

                # Update legend
                handles, labels = ax_select.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_select.legend(by_label.values(), by_label.keys())
                fig_select.canvas.draw()

        # Connect click event
        cid = fig_select.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

        # Process Peak Selection (Phase Shift)
        if selections['peak'] is None:
            print("No point selected. Using default phase shift of 0°.")
            self.phase_shift_deg = 0
            self.max_deflection_idx = None
        else:
            selected_time_s = selections['peak'] / 1e6
            phase_shift_rad = (np.pi / 2) - (2 * np.pi * self.excitation_freq_hz * selected_time_s)
            self.phase_shift_deg = np.degrees(phase_shift_rad)
            # Normalize to [-180, 180] range
            self.phase_shift_deg = ((self.phase_shift_deg + 180) % 360) - 180
            print(f"Calculated phase shift: {self.phase_shift_deg:.2f}°")

            # Store the index of the maximum deflection point for deformation calculation
            self.max_deflection_idx = (self.data['Time_us'] - selections['peak']).abs().idxmin()
            print(f"Maximum deflection index stored: {self.max_deflection_idx}")

        # Process Contact Point Selection
        if selections['contact'] is not None:
            self.contact_point_idx = (self.data['Time_us'] - selections['contact']).abs().idxmin()
            print(f"Contact point index stored: {self.contact_point_idx}")
        else:
            self.contact_point_idx = None
            print("No contact point selected.")

    def set_phase_shift(self, phase_shift_deg):
        """
        Manually set the phase shift in degrees.

        Args:
            phase_shift_deg (float): Phase shift in degrees
        """
        self.phase_shift_deg = phase_shift_deg
        print(f"Phase shift set to: {self.phase_shift_deg:.2f}°")

    def remove_dc_offset_interactively(self):
        """
        Interactively select a region to calculate and remove DC offset from deflection.
        The average deflection in the selected region is subtracted from the entire signal.
        """
        print("Please click on TWO points on the deflection-time plot to select the region for DC offset calculation.")
        print("The average deflection in this region will be subtracted from the entire signal.")
        print("Close the plot window after selecting both points.")

        # Create figure for selection
        fig_select, ax_select = plt.subplots(figsize=(12, 4))
        ax_select.plot(self.data['Time_us'], self.data['Deflection_nm'],
                      label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax_select.set_title('Deflection vs Time (Click to select region for DC offset)')
        ax_select.set_xlabel('Time (µs)')
        ax_select.set_ylabel('Deflection (nm)')
        ax_select.grid(True)
        ax_select.legend()

        # Store selected points
        selected_points = []

        def onclick(event):
            if event.inaxes == ax_select and event.xdata is not None and len(selected_points) < 2:
                # Find the nearest index
                time_us = event.xdata
                idx = (self.data['Time_us'] - time_us).abs().idxmin()
                selected_points.append(idx)

                # Draw vertical line at selected point
                color = 'red' if len(selected_points) == 1 else 'orange'
                ax_select.axvline(x=self.data.loc[idx, 'Time_us'], color=color,
                                         linestyle='--', linewidth=2,
                                         label=f'Point {len(selected_points)}: idx={idx}')
                ax_select.legend()
                fig_select.canvas.draw()
                print(f"Point {len(selected_points)} selected: Index={idx}, Time={self.data.loc[idx, 'Time_us']:.2f} µs")

                if len(selected_points) == 2:
                    print("Two points selected. You can close the window now.")

        # Connect click event
        fig_select.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

        # Validate selection
        if len(selected_points) < 2:
            print("Warning: Less than 2 points selected. DC offset not removed.")
            return

        # Sort indices
        idx1, idx2 = sorted(selected_points)

        # Calculate offset
        region_data = self.data.loc[idx1:idx2, 'Deflection_nm']
        dc_offset = region_data.mean()

        print(f"\nSelected range: Index {idx1} to {idx2}")
        print(f"Calculated DC Offset: {dc_offset:.6f} nm")

        # Apply offset
        self.data['Deflection_nm'] -= dc_offset
        print("DC Offset removed from Deflection data.")

        # Update Separation if it exists
        if 'Separation' in self.data.columns:
             self.data['Separation'] = self.data['Excitation'] - self.data['Deflection_nm']
             print("Separation updated.")
        
        # Update Force if it exists
        if self.spring_constant_N_m is not None:
            self.calculate_force()

    def generate_excitation_signal(self):
        """
        Generate the excitation signal with the current phase shift and calculate separation.
        Separation is defined as Excitation - Deflection.
        """
        # Generate time array in seconds
        time_s = self.data['Index'] * self.sampling_period_s

        # Convert phase shift to radians
        phase_shift_rad = np.radians(self.phase_shift_deg)

        if 'Excitation' in self.data.columns: # Avoid re-calculation
            return

        # Generate sinusoidal excitation signal with phase shift
        self.data['Excitation'] = self.excitation_amp * np.sin(
            2 * np.pi * self.excitation_freq_hz * time_s + phase_shift_rad
        )

        # Calculate Separation
        self.data['Separation'] = self.data['Excitation'] - self.data['Deflection_nm']

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
            'Deflection_nm': self.data['Deflection_nm'],
            'Excitation': self.data['Excitation'],
            'Separation': self.data['Separation']
        })

        # Determine output path
        if output_path is None:
            export_filename = f"{self.file_name_no_ext}_processed.csv"
            output_path = os.path.join(self.base_dir, export_filename)

        # Export to CSV
        export_data.to_csv(output_path, index=False, header=False)
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

        col_name, signal_data, unit_label = self._get_signal_info()

        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

        # Plot 1: Signal vs Time
        ax1.plot(self.data['Time_us'], signal_data,
                label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax1.set_title(f'{plot_title} - {unit_label} vs Time')
        ax1.set_xlabel('Time (µs)')
        ax1.set_ylabel(unit_label)
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Excitation vs Time
        ax2.plot(self.data['Time_us'], self.data['Excitation'], linewidth=1.1, color='red')
        ax2.set_title(f'{plot_title} - Excitation vs Time')
        ax2.set_xlabel('Time (µs)')
        ax2.set_ylabel('Excitation (nm)')
        ax2.grid(True)

        # Plot 3: Signal vs Separation
        ax3.plot(self.data['Separation'], signal_data, linewidth=1.1, color='green')
        ax3.set_title(f'{plot_title} - {unit_label} vs Separation')
        ax3.set_xlabel('Separation (nm)')
        ax3.set_ylabel(unit_label)
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
        col_name, signal_data, unit_label = self._get_signal_info()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.data['Time_us'], signal_data,
               label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax.set_title(f'{unit_label} vs Time')
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel(unit_label)
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

        col_name, signal_data, unit_label = self._get_signal_info()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(self.data['Separation'], signal_data, linewidth=1.1, color='green')
        ax.set_title(f'{unit_label} vs Separation (Hysteresis)')
        ax.set_xlabel('Separation (nm)')
        ax.set_ylabel(unit_label)
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

        col_name, signal_data, unit_label = self._get_signal_info()
        stats = {
            f'{col_name}_mean': signal_data.mean(),
            f'{col_name}_std': signal_data.std(),
            f'{col_name}_min': signal_data.min(),
            f'{col_name}_max': signal_data.max(),
            f'{col_name}_range': signal_data.max() - signal_data.min(),
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
        col_name, signal_data, unit_label = self._get_signal_info()
        
        print(f"Please click on TWO points on the {unit_label}-time plot to select the range for slope calculation.")
        print("Close the plot window after selecting both points.")

        # Create figure for selection
        fig_select, ax_select = plt.subplots(figsize=(12, 4))
        ax_select.plot(self.data['Time_us'], signal_data,
                      label=self.file_name_no_ext, linewidth=1.1, color='blue')
        ax_select.set_title(f'{unit_label} vs Time (Click to select TWO points for slope calculation)')
        ax_select.set_xlabel('Time (µs)')
        ax_select.set_ylabel(unit_label)
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
        Calculate the slope of the deflection vs separation plot between two indices
        using least squares linear regression with zero intercept (through origin).

        Args:
            idx1 (int, optional): Starting index. If None and interactive=True,
                                 will prompt user to select interactively.
            idx2 (int, optional): Ending index. If None and interactive=True,
                                 will prompt user to select interactively.
            interactive (bool): If True and indices not provided, prompts interactive selection.

        Returns:
            dict: Dictionary containing slope (zero intercept), R-squared, and other fit parameters
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

        col_name, signal_data, unit_label = self._get_signal_info()

        # Extract data for the selected range
        separation_range = self.data.loc[idx1:idx2, 'Separation'].values
        signal_range = self.data.loc[idx1:idx2, col_name].values

        # Perform least squares linear fit with zero intercept: deflection = slope * separation
        # For zero-intercept regression, slope = sum(x*y) / sum(x^2)
        slope = np.sum(separation_range * signal_range) / np.sum(separation_range**2)
        intercept = 0  # Forced to zero

        # Calculate fitted values and residuals
        signal_fit = slope * separation_range
        residuals = signal_range - signal_fit
        ss_res = np.sum(residuals**2)  # Residual sum of squares
        ss_tot = np.sum(signal_range**2)  # Total sum of squares (for zero-intercept model)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate standard error of the slope (for zero-intercept model)
        n = len(excitation_range)
        if n > 1 and np.sum(separation_range**2) > 0:
            std_error = np.sqrt(ss_res / (n - 1)) / np.sqrt(np.sum(separation_range**2))
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
            'separation_range': separation_range,
            'signal_range': signal_range,
            'signal_fit': signal_fit,
            'unit_label': unit_label
        }

        # Print results
        print("\n" + "="*60)
        print("SLOPE CALCULATION RESULTS (Zero-Intercept Model)")
        print("="*60)
        print(f"Index range: {idx1} to {idx2} ({n} points)")
        print(f"Slope: {slope:.6e} {unit_label.split(' ')[0]}/nm")
        print(f"Intercept: {intercept:.6f} nm (forced to zero)")
        print(f"R-squared: {r_squared:.6f}")
        print(f"Standard error: {std_error:.6e}")
        print("="*60 + "\n")

        return results

    def calculate_slope_with_intercept(self, idx1=None, idx2=None, interactive=True):
        """
        Calculate the slope and intercept of the deflection vs separation plot
        using standard least squares linear regression (y = mx + c).

        Args:
            idx1 (int, optional): Starting index.
            idx2 (int, optional): Ending index.
            interactive (bool): If True and indices not provided, prompts interactive selection.

        Returns:
            dict: Dictionary containing slope, intercept, R-squared, and other fit parameters.
        """
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        if idx1 is None or idx2 is None:
            if interactive:
                idx1, idx2 = self.select_indices_for_slope()
                if idx1 is None or idx2 is None:
                    print("Error: Invalid indices selected.")
                    return None
            else:
                print("Error: Indices must be provided when interactive=False")
                return None

        col_name, signal_data, unit_label = self._get_signal_info()

        # Use .values to get numpy arrays for efficient calculation
        separation_range = self.data.loc[idx1:idx2, 'Separation'].values
        signal_range = self.data.loc[idx1:idx2, col_name].values
        n = len(separation_range)

        # Implement the standard linear regression formulas directly
        # slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        sum_xy = np.sum(separation_range * signal_range)
        sum_x = np.sum(separation_range)
        sum_y = np.sum(signal_range)
        sum_x_sq = np.sum(separation_range**2)

        print("DEBUGGGGGGGGG")
        print(n * sum_xy, sum_x * sum_y, n * sum_x_sq, sum_x**2)
        print("SUM X")
        print(sum_x)
        print("SUM Y")
        print(sum_y)
        print("SUM xy")
        print(sum_xy)
        print("SUM x^2")
        print(sum_x_sq)
        print("DEBUGGGGGGGGG")
        print(signal_range)
        

        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x_sq - sum_x**2

        slope = numerator / denominator if denominator != 0 else 0

        # intercept = mean(y) - slope * mean(x)
        intercept = (sum_y / n) - slope * (sum_x / n)

        # Calculate fitted values and R-squared
        signal_fit = slope * separation_range + intercept
        ss_res = np.sum((signal_range - signal_fit)**2)
        ss_tot = np.sum((signal_range - np.mean(signal_range))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'idx1': idx1,
            'idx2': idx2,
            'num_points': n,
            'separation_range': separation_range.flatten(),
            'signal_range': signal_range,
            'signal_fit': signal_fit,
            'unit_label': unit_label
        }

        print("\n" + "="*60)
        print("SLOPE CALCULATION RESULTS (With Intercept Model)")
        print("="*60)
        print(f"Index range: {idx1} to {idx2} ({len(separation_range)} points)")
        print(f"Slope (m): {slope:.6e} {unit_label.split(' ')[0]}/nm")
        print(f"Intercept: {intercept:.6f} nm")
        print(f"R-squared: {r_squared:.6f}")
        print("="*60 + "\n")

        return results

    def plot_slope_fit(self, slope_results):
        """
        Visualize the slope fit on the deflection vs separation plot.

        Args:
            slope_results (dict): Results dictionary from calculate_slope()
        """
        if slope_results is None:
            print("Error: No slope results to plot.")
            return

        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        col_name, signal_data, unit_label = self._get_signal_info()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Full deflection vs separation with highlighted region
        ax1.plot(self.data['Separation'], signal_data,
                linewidth=1.1, color='lightgray', alpha=0.5, label='Full data')
        ax1.plot(slope_results['separation_range'], slope_results['signal_range'],
                linewidth=2, color='blue', marker='o', markersize=3, label='Selected range')
        ax1.set_title(f'{unit_label} vs Separation (Full View)')
        ax1.set_xlabel('Separation (nm)')
        ax1.set_ylabel(unit_label)
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Zoomed view with linear fit
        ax2.scatter(slope_results['separation_range'], slope_results['signal_range'],
                   color='blue', alpha=0.6, s=20, label='Data points')
        ax2.plot(slope_results['separation_range'], slope_results['signal_fit'],
                color='red', linewidth=2, linestyle='--', label='Linear fit')
        ax2.set_title(f'Linear Fit (Slope = {slope_results["slope"]:.4f}, R² = {slope_results["r_squared"]:.4f})')
        title = f'Linear Fit (Slope = {slope_results["slope"]:.4f}, R² = {slope_results["r_squared"]:.4f})'
        ax2.set_title(title)
        ax2.set_xlabel('Separation (nm)')
        ax2.set_ylabel(unit_label)
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

    def calculate_deformation(self, percentage=15):
        """
        Calculate the deformation based on the change in separation between the
        maximum deflection point and a specified percentage of the absolute maximum deflection.

        Args:
            percentage (float): The percentage of absolute max deflection to use as reference (default 15).
        Returns:
            dict: Dictionary containing deformation value and related information
        """
        # Ensure excitation signal is generated
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        col_name, signal_data, unit_label = self._get_signal_info()

        # 1. Find the point of Maximum Deflection/Force
        # We use the maximum value of the signal
        max_idx = signal_data.idxmax()
        val_max = signal_data[max_idx]

        # 2. Calculate the target deflection value (percentage of absolute max)
        target_val = (percentage / 100.0) * val_max

        # 3. Find the index corresponding to this target value on the approach curve
        # We look at data points before the maximum deflection
        approach_mask = self.data.index <= max_idx
        approach_data = signal_data.loc[approach_mask]
        # Find index with minimum absolute difference to target value
        idx_percentage = (approach_data - target_val).abs().idxmin()

        # 4. Calculate Deformation: Change in Separation between these two points
        separation_at_max = self.data.loc[max_idx, 'Separation']
        separation_at_percentage = self.data.loc[idx_percentage, 'Separation']
        
        deformation = abs(separation_at_max - separation_at_percentage)
        self.deformation = deformation

        # Get signal values for plotting
        val_percentage = signal_data[idx_percentage]

        # Prepare results
        results = {
            'deformation': deformation,
            'max_idx': max_idx,
            'percentage_idx': idx_percentage,
            'val_max': val_max,
            'val_percentage': val_percentage,
            'percentage': percentage,
            'unit_label': unit_label,
            'time_max': self.data.loc[max_idx, 'Time_us'],
            'time_percentage': self.data.loc[idx_percentage, 'Time_us'],
            'separation_max': separation_at_max,
            'separation_percentage': separation_at_percentage
        }

        # Print results
        print("\n" + "="*60)
        print("DEFORMATION CALCULATION RESULTS")
        print("="*60)
        print(f"Maximum {unit_label.split(' ')[0]} point:")
        print(f"  Index: {max_idx}")
        print(f"  Time: {results['time_max']:.2f} µs")
        print(f"  {unit_label.split(' ')[0]}: {val_max:.4f}")
        print(f"  Separation: {separation_at_max:.4f} nm")
        print(f"\nReference point ({percentage}% of absolute max {unit_label.split(' ')[0]}):")
        print(f"  Index: {idx_percentage}")
        print(f"  Time: {results['time_percentage']:.2f} µs")
        print(f"  {unit_label.split(' ')[0]}: {val_percentage:.4f}")
        print(f"  Separation: {separation_at_percentage:.4f} nm")
        print(f"\nCalculation: |Separation(Max) - Separation({percentage}%)|")
        print(f"DEFORMATION: {deformation:.4f} nm")
        print("="*60 + "\n")

        return results

    def calculate_adhesion(self):
        """
        Calculate the adhesion, defined as the minimum value of the signal
        (Force or Deflection) in the cycle.

        Returns:
            dict: Dictionary containing adhesion value and related information
        """
        # Ensure excitation signal is generated
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        col_name, signal_data, unit_label = self._get_signal_info()

        # Find minimum value (Adhesion)
        min_idx = signal_data.idxmin()
        min_value = signal_data[min_idx]
        separation_at_min = self.data.loc[min_idx, 'Separation']

        results = {
            'min_idx': min_idx,
            'adhesion_value': min_value,
            'separation_at_min': separation_at_min,
            'unit_label': unit_label
        }

        print("\n" + "="*60)
        print("ADHESION CALCULATION RESULTS")
        print("="*60)
        print(f"Minimum {unit_label.split(' ')[0]} (Adhesion):")
        print(f"  Index: {min_idx}")
        print(f"  Value: {min_value:.4f}")
        print(f"  Separation: {separation_at_min:.4f} nm")
        print("="*60 + "\n")

        return results

    def plot_adhesion(self, adhesion_results=None):
        """
        Visualize the adhesion (minimum force point) on the signal vs separation plot.

        Args:
            adhesion_results (dict, optional): Results from calculate_adhesion().
        """
        if adhesion_results is None:
            adhesion_results = self.calculate_adhesion()

        col_name, signal_data, unit_label = self._get_signal_info()

        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot Hysteresis
        ax.plot(self.data['Separation'], signal_data,
                linewidth=1.1, color='green', label='Hysteresis curve')

        # Mark Adhesion point
        ax.plot(adhesion_results['separation_at_min'],
                adhesion_results['adhesion_value'],
                'ro', markersize=10, label='Adhesion (Min Point)')

        # Annotation
        # Extract unit from label "Force (nN)" -> "nN"
        unit = unit_label.split('(')[-1].replace(')', '') if '(' in unit_label else ""

        ax.annotate(f"Adhesion\n{adhesion_results['adhesion_value']:.4f} {unit}",
                    xy=(adhesion_results['separation_at_min'], adhesion_results['adhesion_value']),
                    xytext=(0, -40), textcoords='offset points',
                    ha='center', color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))

        ax.set_title(f'{unit_label} vs Separation - Adhesion')
        ax.set_xlabel('Separation (nm)')
        ax.set_ylabel(unit_label)
        ax.grid(True)
        ax.legend()

        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(self.base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"{self.file_name_no_ext}_adhesion.png")
        plt.savefig(save_path, dpi=300)
        print(f"Adhesion plot saved to '{save_path}'")

        plt.show()

    def calculate_young_modulus_dmt(self, high_factor_pct=90, low_range_pct=30, tip_radius_nm=None,
                                    sample_poisson_ratio=0.3, cantilever_poisson_ratio=0.17,
                                    cantilever_E_GPa=169.0, neglect_cantilever_effect=True):
        """
        Calculate Young's Modulus using the DMT model fit on the retraction curve.
        Equation: Force = 4/3 * E_reduced * sqrt(R * d^3) + Adhesion
        
        Then calculates E_sample using:
        1/E_reduced = (1-v_sample^2)/E_sample + (1-v_cantilever^2)/E_cantilever

        Args:
            high_factor_pct (float): Percentage of the absolute peak force to define the upper fit limit (default 90).
            low_range_pct (float): Percentage of the full range (relative to adhesion) to define the lower fit limit (default 30).
            tip_radius_nm (float): Tip radius in nm.
            sample_poisson_ratio (float): Poisson's ratio of the sample (default 0.3).
            cantilever_poisson_ratio (float): Poisson's ratio of the cantilever tip (default 0.17).
            cantilever_E_GPa (float): Young's Modulus of the cantilever tip in GPa (default 169.0).
            neglect_cantilever_effect (bool): If True, neglects the cantilever term in the elasticity equation (default True).

        Returns:
            dict: Results including calculated E_sample and E_reduced.
        """
        if tip_radius_nm is None or tip_radius_nm <= 0:
            print("Error: Valid tip_radius_nm is required for DMT calculation.")
            return None

        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        col_name, signal_data, unit_label = self._get_signal_info()

        # 1. Identify Region (Retraction)
        max_idx = signal_data.idxmax()
        
        # Find Adhesion (Minimum in retraction) - search after peak
        full_retract_data = signal_data.loc[max_idx:]
        min_idx = full_retract_data.idxmin()
        
        # Define the specific retraction segment (Peak -> Adhesion)
        retract_data = signal_data.loc[max_idx:min_idx]
        retract_separation = self.data.loc[max_idx:min_idx, 'Separation']

        val_peak = signal_data[max_idx]
        val_adhesion = signal_data[min_idx]
        
        # Define Range
        val_range = val_peak - val_adhesion

        # Define Thresholds based on user specification
        # High: 90% of Peak Force (absolute)
        threshold_high = val_peak * (high_factor_pct / 100.0)
        # Low: 30% of Range (relative to adhesion)
        threshold_low = val_adhesion + val_range * (low_range_pct / 100.0)

        # Select Data Points
        mask = (retract_data <= threshold_high) & (retract_data >= threshold_low)
        
        if not mask.any():
            print("Error: No data points found in the specified DMT fitting range.")
            return None

        y_fit_data = retract_data[mask] # Force/Signal
        x_raw_separation = retract_separation[mask]

        # Prepare for Fitting
        # DMT Model: F = 4/3 * E * sqrt(R * delta^3) + F_adh
        # Linear form: (F - F_adh) = (4/3 * E * sqrt(R)) * delta^1.5
        # We assume delta = 0 at the adhesion point (pull-off), so we shift separation.
        separation_at_adhesion = self.data.loc[min_idx, 'Separation']
        delta = x_raw_separation - separation_at_adhesion
        
        # Ensure delta is positive (it should be, as we are 'above' the pull-off point)
        delta = delta.abs() 

        # X axis for regression: sqrt(R * delta^3) = sqrt(R) * delta^1.5
        X = np.sqrt(tip_radius_nm * (delta ** 3))
        
        # Y axis for regression: Force - Adhesion
        Y = y_fit_data - val_adhesion

        # Perform Linear Regression (Force through origin)
        # Slope m = sum(X*Y) / sum(X^2)
        slope = np.sum(X * Y) / np.sum(X**2)
        
        # Calculate E_reduced
        # m = 4/3 * E_reduced  => E_reduced = 3/4 * m
        E_reduced = 0.75 * slope
        
        # Calculate E_sample
        E_sample = E_reduced # Default if not using force or calculation fails
        unit = "GPa" if self.use_force else "arbitrary"
        
        if self.use_force:
            # E_reduced is in GPa
            term_cantilever = 0
            if not neglect_cantilever_effect:
                if cantilever_E_GPa is None or cantilever_E_GPa <= 0:
                     print("Warning: Invalid cantilever E provided. Neglecting cantilever effect.")
                else:
                     term_cantilever = (1 - cantilever_poisson_ratio**2) / cantilever_E_GPa
            
            # 1/E_red = (1-v_s^2)/E_s + term_c
            # (1-v_s^2)/E_s = 1/E_red - term_c
            # E_s = (1-v_s^2) / (1/E_red - term_c)
            
            denom = (1.0 / E_reduced) - term_cantilever
            if denom <= 0:
                 print("Warning: Calculated E_sample is negative or infinite (cantilever term too large). Check inputs.")
                 E_sample = np.nan
            else:
                 E_sample = (1 - sample_poisson_ratio**2) / denom

        # Calculate R-squared for the linear fit
        Y_pred = slope * X
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - np.mean(Y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print("\n" + "="*60)
        print("DMT MODULUS CALCULATION")
        print("="*60)
        print(f"Fitting Range: {high_factor_pct}% of Peak to {low_range_pct}% of Range")
        print(f"Points used: {len(Y)}")
        print(f"Calculated E_reduced: {E_reduced:.6f} {unit}")
        if self.use_force:
            print(f"Calculated E_sample: {E_sample:.6f} {unit}")
            print(f"  (Sample Poisson ratio: {sample_poisson_ratio})")
            if not neglect_cantilever_effect:
                print(f"  (Cantilever correction applied: E={cantilever_E_GPa} GPa, v={cantilever_poisson_ratio})")
            else:
                print("  (Cantilever effect neglected)")
        print(f"Fit R-squared: {r_squared:.6f}")
        print("="*60 + "\n")

        return {
            'E_sample': E_sample,
            'E_reduced': E_reduced,
            'E_DMT': E_reduced, # For backward compatibility
            'unit': unit,
            'slope': slope,
            'r_squared': r_squared,
            'X': X,
            'Y': Y,
            'Y_fit': Y_pred,
            'separation_fit': x_raw_separation,
            'signal_fit': y_fit_data,
            'high_factor_pct': high_factor_pct,
            'low_range_pct': low_range_pct
        }

    def plot_dmt_fit(self, dmt_results):
        """
        Visualize the DMT model fit.

        Args:
            dmt_results (dict): Results from calculate_young_modulus_dmt.
        """
        if dmt_results is None:
            return

        col_name, signal_data, unit_label = self._get_signal_info()
        unit_short = unit_label.split('(')[-1].replace(')', '') if '(' in unit_label else ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Signal vs Separation (Physical Space)
        # Plot full data
        ax1.plot(self.data['Separation'], signal_data,
                 color='lightgray', label='Full Cycle', linewidth=1)

        # Highlight fitted region
        ax1.scatter(dmt_results['separation_fit'], dmt_results['signal_fit'],
                    color='blue', s=10, label='Fitted Region')

        ax1.set_title(f'DMT Fit Region (High: {dmt_results["high_factor_pct"]}% Peak, Low: {dmt_results["low_range_pct"]}% Range)')
        ax1.set_xlabel('Separation (nm)')
        ax1.set_ylabel(unit_label)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Linearized Space (Hertz/DMT transform)
        # X = sqrt(R * delta^3), Y = F - F_adh
        ax2.scatter(dmt_results['X'], dmt_results['Y'],
                    color='blue', s=10, alpha=0.5, label='Data')
        ax2.plot(dmt_results['X'], dmt_results['Y_fit'],
                 color='red', linestyle='--', linewidth=2, label=f'Fit (R²={dmt_results["r_squared"]:.4f})')

        title_str = f'DMT Linearization\nE_sample = {dmt_results["E_sample"]:.4f} {dmt_results["unit"]}'
        if "E_reduced" in dmt_results:
             title_str += f'\n(E_reduced = {dmt_results["E_reduced"]:.4f} {dmt_results["unit"]})'
        ax2.set_title(title_str)
        ax2.set_xlabel(r'$\sqrt{R \cdot \delta^3}$ ($nm^{1.5}$)')
        ax2.set_ylabel(f'Signal - Adhesion ({unit_short})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(self.base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"{self.file_name_no_ext}_dmt_fit.png")
        plt.savefig(save_path, dpi=300)
        print(f"DMT fit plot saved to '{save_path}'")

        plt.show()

    def plot_deformation(self, deformation_results=None):
        """
        Visualize the deformation measurement on the deflection vs time plot
        and deflection vs separation plot.

        Args:
            deformation_results (dict, optional): Results from calculate_deformation().
                                                 If None, will call calculate_deformation().
        """
        if deformation_results is None:
            deformation_results = self.calculate_deformation()

        if deformation_results is None:
            print("Error: No deformation results to plot.")
            return

        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        col_name, signal_data, unit_label = self._get_signal_info()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Deflection vs Time with marked points
        ax1.plot(self.data['Time_us'], signal_data,
                linewidth=1.1, color='blue', label=unit_label.split(' ')[0])

        # Mark maximum deflection point
        max_idx = deformation_results['max_idx']
        ax1.plot(self.data.loc[max_idx, 'Time_us'],
                self.data.loc[max_idx, col_name],
                'ro', markersize=10, label=f'Max Deflection Point')

        # Mark minimum deflection point
        p_idx = deformation_results['percentage_idx']
        ax1.plot(self.data.loc[p_idx, 'Time_us'],
                self.data.loc[p_idx, col_name],
                'go', markersize=10, label=f'{deformation_results["percentage"]}% of Max Point')

        ax1.set_title(f'{unit_label} vs Time - Deformation Measurement Points')
        ax1.set_xlabel('Time (µs)')
        ax1.set_ylabel(unit_label)
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Deflection vs Separation with marked points and deformation arrow
        ax2.plot(self.data['Separation'], signal_data,
                linewidth=1.1, color='green', label='Hysteresis curve')

        # Mark maximum deflection point
        sep_max = deformation_results['separation_max']
        ax2.plot(sep_max,
                deformation_results['val_max'],
                'ro', markersize=10, label='Max Point')

        # Mark minimum deflection point
        sep_p = deformation_results['separation_percentage']
        ax2.plot(sep_p,
                deformation_results['val_percentage'],
                'go', markersize=10, label=f'{deformation_results["percentage"]}% Point')

        # Draw arrow showing deformation
        # Note: Deformation is calculated from Deflection, but plotted on Signal axis (which might be Force)
        ax2.annotate('',
                    xy=(sep_max,
                        (deformation_results['val_max'] + deformation_results['val_percentage']) / 2),
                    xytext=(sep_p,
                           (deformation_results['val_max'] + deformation_results['val_percentage']) / 2),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))

        # Add text showing deformation value
        mid_separation = (sep_max + sep_p) / 2
        mid_signal = (deformation_results['val_max'] + deformation_results['val_percentage']) / 2
        ax2.text(mid_separation, mid_signal,
                f"Deformation\n{deformation_results['deformation']:.4f} nm",
                ha='center', va='bottom', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_title(f'{unit_label} vs Separation - Deformation = {deformation_results["deformation"]:.4f} nm')
        ax2.set_xlabel('Separation (nm)')
        ax2.set_ylabel(unit_label)
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(self.base_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"{self.file_name_no_ext}_deformation.png")
        plt.savefig(save_path, dpi=300)
        print(f"Deformation plot saved to '{save_path}'")

        plt.show()

    def calculate_hysteresis_area(self):
        """
        Calculate the area under the deflection-excitation curve for forward
        and backward paths, and compute their difference.

        The forward path is from start to max_deflection_idx.
        The backward path is from max_deflection_idx to end.

        Returns:
            dict: Dictionary containing forward area, backward area, and net area difference
        """
        # Ensure excitation signal is generated
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        # Check if maximum deflection index has been set
        if self.max_deflection_idx is None:
            print("Error: Maximum deflection point not selected. Please run select_phase_interactively() first.")
            return None

        col_name, signal_data, unit_label = self._get_signal_info()

        # Split data into forward and backward segments
        forward_excitation = self.data.loc[:self.max_deflection_idx, 'Excitation'].values
        forward_signal = self.data.loc[:self.max_deflection_idx, col_name].values

        backward_excitation = self.data.loc[self.max_deflection_idx:, 'Excitation'].values
        backward_signal = self.data.loc[self.max_deflection_idx:, col_name].values

        # Calculate areas using Right Riemann sum
        # Area = integral of deflection with respect to excitation
        forward_area = np.sum(forward_signal[1:] * np.diff(forward_excitation))
        backward_area = np.sum(backward_signal[1:] * np.diff(backward_excitation))

        area_unit = "nm²"
        if self.use_force:
            # Convert from aJ (nN*nm) to eV
            # 1 aJ = 6.241509 eV
            conversion_factor = 6.241509
            forward_area *= conversion_factor
            backward_area *= conversion_factor
            area_unit = "eV"

        # Net area is the difference (this represents hysteresis/energy dissipation)
        net_area = abs(forward_area - backward_area)

        # Prepare results
        results = {
            'forward_area': forward_area,
            'backward_area': backward_area,
            'net_area': net_area,
            'area_difference': forward_area - backward_area,
            'max_deflection_idx': self.max_deflection_idx,
            'forward_excitation': forward_excitation,
            'forward_signal': forward_signal,
            'backward_excitation': backward_excitation,
            'backward_signal': backward_signal,
            'unit_label': unit_label
        }

        # Print results
        print("\n" + "="*60)
        print("HYSTERESIS AREA CALCULATION RESULTS (vs Excitation)")
        print("="*60)
        print(f"Split point: Index {self.max_deflection_idx} (max deflection)")
        print(f"\nForward path (0 to {self.max_deflection_idx}):")
        print(f"  Number of points: {len(forward_excitation)}")
        print(f"  Area: {forward_area:.6e} {area_unit}")
        print(f"\nBackward path ({self.max_deflection_idx} to {len(self.data)-1}):")
        print(f"  Number of points: {len(backward_excitation)}")
        print(f"  Area: {backward_area:.6e} {area_unit}")
        print(f"\nArea difference: {forward_area - backward_area:.6e} {area_unit}")
        print(f"Net area (absolute): {net_area:.6e} {area_unit}")
        print("="*60 + "\n")

        return results

    def plot_hysteresis_area(self, area_results=None, save=True):
        """
        Visualize the hysteresis curve with forward and backward areas filled
        and annotated with their values.

        Args:
            area_results (dict, optional): Results from calculate_hysteresis_area().
                                          If None, will call calculate_hysteresis_area().
            save (bool): Whether to save the plot to file
        """
        if area_results is None:
            area_results = self.calculate_hysteresis_area()

        if area_results is None:
            print("Error: No area results to plot.")
            return

        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        col_name, signal_data, unit_label = self._get_signal_info()
        area_unit = "nm²" if not self.use_force else "eV"

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the full hysteresis curve
        ax.plot(self.data['Excitation'], signal_data,
               linewidth=2, color='black', label='Hysteresis curve', zorder=3)

        # Fill forward area (start to max_deflection_idx)
        ax.fill_between(area_results['forward_excitation'],
                       area_results['forward_signal'],
                       alpha=0.3, color='blue', label='Forward area')

        # Fill backward area (max_deflection_idx to end)
        ax.fill_between(area_results['backward_excitation'],
                       area_results['backward_signal'],
                       alpha=0.3, color='red', label='Backward area')

        # Mark the split point (max deflection)
        max_idx = area_results['max_deflection_idx']
        ax.plot(self.data.loc[max_idx, 'Excitation'],
               self.data.loc[max_idx, col_name],
               'go', markersize=12, label='Split point (max deflection)', zorder=4)

        # Add text annotations for areas
        # Find good positions for text annotations
        forward_mid_idx = len(area_results['forward_excitation']) // 2
        forward_text_x = area_results['forward_excitation'][forward_mid_idx]
        forward_text_y = area_results['forward_signal'][forward_mid_idx]

        backward_mid_idx = len(area_results['backward_excitation']) // 2
        backward_text_x = area_results['backward_excitation'][backward_mid_idx]
        backward_text_y = area_results['backward_signal'][backward_mid_idx]

        ax.text(forward_text_x, forward_text_y,
               f"Forward Area\n{area_results['forward_area']:.3e} {area_unit}",
               ha='center', va='center', fontsize=10, color='blue',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue'),
               zorder=5)

        ax.text(backward_text_x, backward_text_y,
               f"Backward Area\n{area_results['backward_area']:.3e} {area_unit}",
               ha='center', va='center', fontsize=10, color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'),
               zorder=5)

        # Add title with net area
        ax.set_title(f'Hysteresis Area Analysis (vs Excitation)\n' +
                    f'Net Area Difference = {area_results["net_area"]:.3e} {area_unit}\n' +
                    f'(|Forward - Backward| = {abs(area_results["area_difference"]):.3e} {area_unit})',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Excitation (nm)', fontsize=11)
        ax.set_ylabel(unit_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()

        if save:
            # Save plot
            plots_dir = os.path.join(self.base_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f"{self.file_name_no_ext}_hysteresis_area.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Hysteresis area plot saved to '{save_path}'")

        plt.show()

    def calculate_reduced_young_modulus(self, slope_results, deformation_results, tip_radius_nm):
        """
        Calculates the reduced Young's Modulus (Er) based on the Hertz model.

        The formula used is: Er = (3/4) * slope / sqrt(R * deformation)

        Args:
            slope_results (dict): The results dictionary from a slope calculation method.
            deformation_results (dict): The results dictionary from calculate_deformation().
            tip_radius_nm (float): The radius of the AFM tip in nanometers (nm).

        Returns:
            dict: A dictionary containing the reduced Young's Modulus (Er) in GPa
                  and the inputs used. Returns None if inputs are invalid.
        """
        if slope_results is None or 'slope' not in slope_results:
            print("Error: Invalid slope results provided for Young's Modulus calculation.")
            return None

        if deformation_results is None or 'deformation' not in deformation_results:
            print("Error: Invalid deformation results provided for Young's Modulus calculation.")
            return None

        if tip_radius_nm <= 0:
            print("Error: Tip radius must be a positive number.")
            return None

        slope = slope_results['slope']  # Dimensionless (nm/nm)
        deformation = deformation_results['deformation']  # in nm

        if deformation <= 0:
            print("Warning: Deformation is zero or negative, cannot calculate Young's Modulus.")
            return None

        # Convert inputs from nanometers to meters for SI unit consistency
        deformation_m = deformation * 1e-9
        tip_radius_m = tip_radius_nm * 1e-9

        # Er = (3/4) * slope / sqrt(R * deformation)
        # The slope is dimensionless. R and deformation are in nm.
        # The result will be in 1/sqrt(nm^2) which is not a pressure unit.
        
        er_unit = "arbitrary units"
        if self.use_force:
            # Slope is in nN/nm = N/m
            # R and deformation are in nm
            # Er = 0.75 * slope / sqrt(R * deformation)
            # Unit analysis: (N/m) / sqrt(nm * nm) = (N/m) / nm = N / (m * 1e-9 m) = 1e9 N/m^2 = GPa
            # So if we use slope in nN/nm and R, def in nm, the result is directly in GPa.
            er_value = (3 / 4) * slope / np.sqrt(tip_radius_nm * deformation)
            er_unit = "GPa"
        else:
            # Slope is dimensionless (nm/nm)
            er_value = (3 / 4) * slope / np.sqrt(tip_radius_m * deformation_m)

        results = {
            'reduced_young_modulus': er_value,
            'slope_used': slope,
            'deformation_used': deformation,
            'tip_radius_nm': tip_radius_nm
        }

        print("\n" + "="*60)
        print("REDUCED YOUNG'S MODULUS CALCULATION")
        print("="*60)
        print(f"Formula: Er = (3/4) * slope / sqrt(R * deformation)")
        print(f"  Slope: {slope:.4f}")
        print(f"  Deformation: {deformation:.4f} nm")
        print(f"  Tip Radius (R): {tip_radius_nm} nm")
        print(f"  Reduced Young's Modulus (Er): {er_value:.6f} [{er_unit}]")
        print("="*60 + "\n")

        return results
