"""
Utility functions for mechanical property measurements.
"""


def parse_voltage(value_str):
    """
    Parses a string with a voltage value and unit (mV, uV, V)
    and returns the value in millivolts (mV).

    Args:
        value_str: String containing voltage value with unit

    Returns:
        float: Voltage value in millivolts, or None if parsing fails

    Examples:
        >>> parse_voltage("5.2mV")
        5.2
        >>> parse_voltage("100uV")
        0.1
        >>> parse_voltage("0.005V")
        5.0
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
