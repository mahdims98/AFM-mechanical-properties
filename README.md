# Mechanical Property Measurements

A Python package for processing and analyzing mechanical property measurement data from deflection measurements.

## Project Structure

```
Code - calculation of mechanical properties/
├── mechanical_properties/          # Main package
│   ├── __init__.py                # Package initialization
│   ├── utils.py                   # Utility functions (parse_voltage)
│   ├── mechanical_property_data.py # Main MechanicalPropertyData class
│   └── plotting.py                # Legacy plotting functions
├── calculate_mechanical_properties.py  # Main script
├── calculate mechanical properties.py  # Original monolithic file (deprecated)
├── Marcos Data/                   # Data directory
└── plots/                         # Output plots directory

```

## Installation

No installation required. Just ensure you have the required dependencies:

```bash
pip install pandas matplotlib numpy
```

## Usage

### Basic Usage

```python
from mechanical_properties import MechanicalPropertyData

# Create a data object
mech_data = MechanicalPropertyData(
    file_path="path/to/data.csv",
    sampling_freq_hz=1e6,
    sensitivity_scaled=0.001,
    excitation_amp=100e-9
)

# Select phase shift interactively
mech_data.select_phase_interactively()

# Generate excitation signal
mech_data.generate_excitation_signal()

# Plot all data
mech_data.plot_all(save=True)
```

### Calculate Slope (NEW!)

```python
# Interactive mode - click two points on deflection-time plot
slope_results = mech_data.calculate_slope(interactive=True)

# Visualize the fit
if slope_results is not None:
    mech_data.plot_slope_fit(slope_results)

# Access slope value
print(f"Slope: {slope_results['slope']:.6e}")
print(f"R-squared: {slope_results['r_squared']:.6f}")
```

### Manual Index Selection

```python
# If you know the index range
slope_results = mech_data.calculate_slope(idx1=100, idx2=200, interactive=False)
```

## Module Overview

### `mechanical_properties.utils`

- **`parse_voltage(value_str)`**: Parses voltage strings with units (mV, uV, V) and converts to millivolts.

### `mechanical_properties.MechanicalPropertyData`

Main class for data processing with the following methods:

#### Data Loading & Management
- `__init__(...)`: Initialize with file path and measurement parameters
- `get_data()`: Get processed data as pandas DataFrame
- `get_statistics()`: Get statistical summary of the data
- `export_to_csv()`: Export processed data to CSV

#### Phase & Signal Generation
- `select_phase_interactively()`: Interactively select phase shift from plot
- `set_phase_shift(phase_deg)`: Manually set phase shift
- `generate_excitation_signal()`: Generate sinusoidal excitation signal

#### Plotting
- `plot_all()`: Create all three plots (deflection vs time, excitation vs time, hysteresis)
- `plot_deflection()`: Plot only deflection vs time
- `plot_hysteresis()`: Plot only deflection vs excitation

#### Slope Analysis (NEW!)
- `select_indices_for_slope()`: Interactively select two points for slope calculation
- `calculate_slope()`: Calculate slope using least squares linear regression
- `plot_slope_fit()`: Visualize the linear fit and selected region

### `mechanical_properties.plot_data`

Legacy function for backwards compatibility with older scripts.

## Slope Calculation Workflow

1. **Select Points**: Click two points on the deflection-time plot to define the range
2. **Extract Data**: The method extracts deflection and excitation data for that index range
3. **Fit Line**: Performs least squares linear regression
4. **Get Results**: Returns slope, intercept, R², standard error, and fitted values
5. **Visualize**: Shows the fit quality with two plots (full view + zoomed fit)

## Output Files

- `*_processed.csv`: Processed data with Index, Deflection, and Excitation columns
- `*_all_plots.png`: Three-panel plot (deflection vs time, excitation vs time, hysteresis)
- `*_slope_fit.png`: Two-panel slope fit visualization

## Notes

- The old monolithic file `calculate mechanical properties.py` is kept for reference but is deprecated
- Use the new modular structure with `calculate_mechanical_properties.py` (underscore) as the entry point
- All plots are automatically saved to the `plots/` subdirectory
