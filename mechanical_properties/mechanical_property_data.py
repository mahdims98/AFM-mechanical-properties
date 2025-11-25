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
        self.max_deflection_idx = None  # Index of maximum deflection point selected by user
        self.deformation = None  # Calculated deformation value

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
            self.max_deflection_idx = None
        else:
            selected_time_s = selected_time['time_us'] / 1e6
            phase_shift_rad = (np.pi / 2) - (2 * np.pi * self.excitation_freq_hz * selected_time_s)
            self.phase_shift_deg = np.degrees(phase_shift_rad)
            # Normalize to [-180, 180] range
            self.phase_shift_deg = ((self.phase_shift_deg + 180) % 360) - 180
            print(f"Calculated phase shift: {self.phase_shift_deg:.2f}°")

            # Store the index of the maximum deflection point for deformation calculation
            self.max_deflection_idx = (self.data['Time_us'] - selected_time['time_us']).abs().idxmin()
            print(f"Maximum deflection index stored: {self.max_deflection_idx}")

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

        # Extract data for the selected range
        excitation_range = self.data.loc[idx1:idx2, 'Excitation'].values
        deflection_range = self.data.loc[idx1:idx2, 'Deflection_mV'].values

        # Perform least squares linear fit with zero intercept: deflection = slope * excitation
        # For zero-intercept regression, slope = sum(x*y) / sum(x^2)
        slope = np.sum(excitation_range * deflection_range) / np.sum(excitation_range**2)
        intercept = 0  # Forced to zero

        # Calculate fitted values and residuals
        deflection_fit = slope * excitation_range
        residuals = deflection_range - deflection_fit
        ss_res = np.sum(residuals**2)  # Residual sum of squares
        ss_tot = np.sum(deflection_range**2)  # Total sum of squares (for zero-intercept model)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Calculate standard error of the slope (for zero-intercept model)
        n = len(excitation_range)
        if n > 1:
            std_error = np.sqrt(ss_res / (n - 1)) / np.sqrt(np.sum(excitation_range**2))
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
        print("SLOPE CALCULATION RESULTS (Zero-Intercept Model)")
        print("="*60)
        print(f"Index range: {idx1} to {idx2} ({n} points)")
        print(f"Slope: {slope:.6e} nm/nm (dimensionless)")
        print(f"Intercept: {intercept:.6f} nm (forced to zero)")
        print(f"R-squared: {r_squared:.6f}")
        print(f"Standard error: {std_error:.6e}")
        print("="*60 + "\n")

        return results

    def calculate_slope_with_intercept(self, idx1=None, idx2=None, interactive=True):
        """
        Calculate the slope and intercept of the deflection vs excitation plot
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

        # Use .values to get numpy arrays for efficient calculation
        excitation_range = self.data.loc[idx1:idx2, 'Excitation'].values
        deflection_range = self.data.loc[idx1:idx2, 'Deflection_mV'].values
        n = len(excitation_range)

        # Implement the standard linear regression formulas directly
        # slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
        sum_xy = np.sum(excitation_range * deflection_range)
        sum_x = np.sum(excitation_range)
        sum_y = np.sum(deflection_range)
        sum_x_sq = np.sum(excitation_range**2)

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
        print(deflection_range)
        

        numerator = n * sum_xy - sum_x * sum_y
        denominator = n * sum_x_sq - sum_x**2

        slope = numerator / denominator if denominator != 0 else 0

        # intercept = mean(y) - slope * mean(x)
        intercept = (sum_y / n) - slope * (sum_x / n)

        # Calculate fitted values and R-squared
        deflection_fit = slope * excitation_range + intercept
        ss_res = np.sum((deflection_range - deflection_fit)**2)
        ss_tot = np.sum((deflection_range - np.mean(deflection_range))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'idx1': idx1,
            'idx2': idx2,
            'num_points': n,
            'excitation_range': excitation_range.flatten(),
            'deflection_range': deflection_range,
            'deflection_fit': deflection_fit
        }

        print("\n" + "="*60)
        print("SLOPE CALCULATION RESULTS (With Intercept Model)")
        print("="*60)
        print(f"Index range: {idx1} to {idx2} ({len(excitation_range)} points)")
        print(f"Slope (m): {slope:.6e} nm/nm (dimensionless)")
        print(f"Intercept: {intercept:.6f} nm")
        print(f"R-squared: {r_squared:.6f}")
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
        title = f'Linear Fit (Slope = {slope_results["slope"]:.4f}, R² = {slope_results["r_squared"]:.4f})'
        ax2.set_title(title)
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

    def calculate_deformation(self):
        """
        Calculate the deformation as the change in excitation between the maximum
        deflection point (selected by user during phase selection) and the minimum
        deflection point (found automatically).

        Returns:
            dict: Dictionary containing deformation value and related information
        """
        # Ensure excitation signal is generated
        if 'Excitation' not in self.data.columns:
            self.generate_excitation_signal()

        # Check if maximum deflection index has been set
        if self.max_deflection_idx is None:
            print("Error: Maximum deflection point not selected. Please run select_phase_interactively() first.")
            return None

        # Find the index of minimum deflection
        min_deflection_idx = self.data['Deflection_mV'][:self.max_deflection_idx].idxmin()

        # Get deflection values at both points
        max_deflection_value = self.data.loc[self.max_deflection_idx, 'Deflection_mV']
        min_deflection_value = self.data.loc[min_deflection_idx, 'Deflection_mV']

        # Get excitation values at both points
        excitation_at_max = self.data.loc[self.max_deflection_idx, 'Excitation']
        excitation_at_min = self.data.loc[min_deflection_idx, 'Excitation']

        # Calculate deformation as the change in excitation
        deformation = abs(excitation_at_max - excitation_at_min)
        self.deformation = deformation

        # Prepare results
        results = {
            'deformation': deformation,
            'max_deflection_idx': self.max_deflection_idx,
            'min_deflection_idx': min_deflection_idx,
            'max_deflection_value': max_deflection_value,
            'min_deflection_value': min_deflection_value,
            'excitation_at_max': excitation_at_max,
            'excitation_at_min': excitation_at_min,
            'deflection_range': max_deflection_value - min_deflection_value,
            'max_time_us': self.data.loc[self.max_deflection_idx, 'Time_us'],
            'min_time_us': self.data.loc[min_deflection_idx, 'Time_us']
        }

        # Print results
        print("\n" + "="*60)
        print("DEFORMATION CALCULATION RESULTS")
        print("="*60)
        print(f"Maximum deflection point (user selected):")
        print(f"  Index: {self.max_deflection_idx}")
        print(f"  Time: {results['max_time_us']:.2f} µs")
        print(f"  Deflection: {max_deflection_value:.4f} nm")
        print(f"  Excitation: {excitation_at_max:.4f} nm")
        print(f"\nMinimum deflection point (auto-detected):")
        print(f"  Index: {min_deflection_idx}")
        print(f"  Time: {results['min_time_us']:.2f} µs")
        print(f"  Deflection: {min_deflection_value:.4f} nm")
        print(f"  Excitation: {excitation_at_min:.4f} nm")
        print(f"\nDeflection range: {results['deflection_range']:.4f} nm")
        print(f"DEFORMATION: {deformation:.4f} nm")
        print("="*60 + "\n")

        return results

    def plot_deformation(self, deformation_results=None):
        """
        Visualize the deformation measurement on the deflection vs time plot
        and deflection vs excitation plot.

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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Deflection vs Time with marked points
        ax1.plot(self.data['Time_us'], self.data['Deflection_mV'],
                linewidth=1.1, color='blue', label='Deflection')

        # Mark maximum deflection point
        max_idx = deformation_results['max_deflection_idx']
        ax1.plot(self.data.loc[max_idx, 'Time_us'],
                self.data.loc[max_idx, 'Deflection_mV'],
                'ro', markersize=10, label=f'Max deflection (user selected)')

        # Mark minimum deflection point
        min_idx = deformation_results['min_deflection_idx']
        ax1.plot(self.data.loc[min_idx, 'Time_us'],
                self.data.loc[min_idx, 'Deflection_mV'],
                'go', markersize=10, label=f'Min deflection (auto-detected)')

        ax1.set_title('Deflection vs Time - Deformation Measurement Points')
        ax1.set_xlabel('Time (µs)')
        ax1.set_ylabel('Deflection (nm)')
        ax1.grid(True)
        ax1.legend()

        # Plot 2: Deflection vs Excitation with marked points and deformation arrow
        ax2.plot(self.data['Excitation'], self.data['Deflection_mV'],
                linewidth=1.1, color='green', label='Hysteresis curve')

        # Mark maximum deflection point
        ax2.plot(deformation_results['excitation_at_max'],
                deformation_results['max_deflection_value'],
                'ro', markersize=10, label='Max deflection point')

        # Mark minimum deflection point
        ax2.plot(deformation_results['excitation_at_min'],
                deformation_results['min_deflection_value'],
                'go', markersize=10, label='Min deflection point')

        # Draw arrow showing deformation
        ax2.annotate('',
                    xy=(deformation_results['excitation_at_max'],
                        (deformation_results['max_deflection_value'] + deformation_results['min_deflection_value']) / 2),
                    xytext=(deformation_results['excitation_at_min'],
                           (deformation_results['max_deflection_value'] + deformation_results['min_deflection_value']) / 2),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))

        # Add text showing deformation value
        mid_excitation = (deformation_results['excitation_at_max'] + deformation_results['excitation_at_min']) / 2
        mid_deflection = (deformation_results['max_deflection_value'] + deformation_results['min_deflection_value']) / 2
        ax2.text(mid_excitation, mid_deflection,
                f"Deformation\n{deformation_results['deformation']:.4f} nm",
                ha='center', va='bottom', fontsize=10, color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_title(f'Deflection vs Excitation - Deformation = {deformation_results["deformation"]:.4f} nm')
        ax2.set_xlabel('Excitation (nm)')
        ax2.set_ylabel('Deflection (nm)')
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

        # Split data into forward and backward segments
        forward_excitation = self.data.loc[:self.max_deflection_idx, 'Excitation'].values
        forward_deflection = self.data.loc[:self.max_deflection_idx, 'Deflection_mV'].values

        backward_excitation = self.data.loc[self.max_deflection_idx:, 'Excitation'].values
        backward_deflection = self.data.loc[self.max_deflection_idx:, 'Deflection_mV'].values

        print("forward deflection", forward_deflection[1:])
        print("forward delta",  np.diff(forward_excitation))
        print("forward multiplication", backward_deflection[1:] * np.diff(backward_excitation))

        # Calculate areas using Right Riemann sum
        # Area = integral of deflection with respect to excitation
        forward_area = np.sum(forward_deflection[1:] * np.diff(forward_excitation))
        backward_area = np.sum(backward_deflection[1:] * np.diff(backward_excitation))

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
            'forward_deflection': forward_deflection,
            'backward_excitation': backward_excitation,
            'backward_deflection': backward_deflection
        }

        # Print results
        print("\n" + "="*60)
        print("HYSTERESIS AREA CALCULATION RESULTS")
        print("="*60)
        print(f"Split point: Index {self.max_deflection_idx} (max deflection)")
        print(f"\nForward path (0 to {self.max_deflection_idx}):")
        print(f"  Number of points: {len(forward_excitation)}")
        print(f"  Area: {forward_area:.6e} nm²")
        print(f"\nBackward path ({self.max_deflection_idx} to {len(self.data)-1}):")
        print(f"  Number of points: {len(backward_excitation)}")
        print(f"  Area: {backward_area:.6e} nm²")
        print(f"\nArea difference: {forward_area - backward_area:.6e} nm²")
        print(f"Net area (absolute): {net_area:.6e} nm²")
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

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the full hysteresis curve
        ax.plot(self.data['Excitation'], self.data['Deflection_mV'],
               linewidth=2, color='black', label='Hysteresis curve', zorder=3)

        # Fill forward area (start to max_deflection_idx)
        ax.fill_between(area_results['forward_excitation'],
                       area_results['forward_deflection'],
                       alpha=0.3, color='blue', label='Forward area')

        # Fill backward area (max_deflection_idx to end)
        ax.fill_between(area_results['backward_excitation'],
                       area_results['backward_deflection'],
                       alpha=0.3, color='red', label='Backward area')

        # Mark the split point (max deflection)
        max_idx = area_results['max_deflection_idx']
        ax.plot(self.data.loc[max_idx, 'Excitation'],
               self.data.loc[max_idx, 'Deflection_mV'],
               'go', markersize=12, label='Split point (max deflection)', zorder=4)

        # Add text annotations for areas
        # Find good positions for text annotations
        forward_mid_idx = len(area_results['forward_excitation']) // 2
        forward_text_x = area_results['forward_excitation'][forward_mid_idx]
        forward_text_y = area_results['forward_deflection'][forward_mid_idx]

        backward_mid_idx = len(area_results['backward_excitation']) // 2
        backward_text_x = area_results['backward_excitation'][backward_mid_idx]
        backward_text_y = area_results['backward_deflection'][backward_mid_idx]

        ax.text(forward_text_x, forward_text_y,
               f"Forward Area\n{area_results['forward_area']:.3e} nm²",
               ha='center', va='center', fontsize=10, color='blue',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='blue'),
               zorder=5)

        ax.text(backward_text_x, backward_text_y,
               f"Backward Area\n{area_results['backward_area']:.3e} nm²",
               ha='center', va='center', fontsize=10, color='red',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'),
               zorder=5)

        # Add title with net area
        ax.set_title(f'Hysteresis Area Analysis\n' +
                    f'Net Area Difference = {area_results["net_area"]:.3e} nm²\n' +
                    f'(|Forward - Backward| = {abs(area_results["area_difference"]):.3e} nm²)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Excitation (nm)', fontsize=11)
        ax.set_ylabel('Deflection (nm)', fontsize=11)
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
