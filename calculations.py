"""
BOVIGENIX - Calculations & Scientific Formulas (utils/calculations.py)
----------------------------------------------------------------------
This module contains domain-specific scientific calculations for dairy cattle
monitoring, including:
1. Temperature-Humidity Index (THI) for environmental heat stress assessment.
2. Cow age calculation from Date of Birth (DOB).
3. Baseline deviation calculations (comparing a cow's daily behavior against
   her personal rolling baseline).

Concepts for Biotechnology Students:
- THI combines ambient temperature and relative humidity into a single number
  representing the thermal comfort / heat stress on cows.
- Cows are homeotherms, but high ambient heat + humidity impairs their ability
  to dissipate heat via evaporative cooling (sweating & panting), reducing milk
  yield, fertility, and dry matter intake.
- CRITICAL BIOLOGICAL RULE: Never use cow body temperature in THI calculations!
  THI measures environmental climatic load, not internal core temperature.
"""

from datetime import datetime, date


def calculate_thi(ambient_temp_c, relative_humidity_pct):
    """
    Calculates the Temperature-Humidity Index (THI) using the standard
    National Research Council (NRC) formula for dairy cattle:

    THI = (1.8 * T + 32) - (0.55 - 0.0055 * RH) * (1.8 * T - 26)

    Parameters:
    - ambient_temp_c (float): Ambient environmental air temperature in Celsius (°C).
    - relative_humidity_pct (float): Relative humidity percentage (0 to 100).

    Returns:
    - float: Calculated THI rounded to 1 decimal place, or None if inputs are invalid.
    """
    if ambient_temp_c is None or relative_humidity_pct is None:
        return None

    try:
        t = float(ambient_temp_c)
        rh = float(relative_humidity_pct)

        # NRC formula
        thi = (1.8 * t + 32.0) - (0.55 - 0.0055 * rh) * (1.8 * t - 26.0)
        return round(thi, 1)
    except (ValueError, TypeError):
        return None


def get_heat_stress_category(thi):
    """
    Categorizes the THI value into standard dairy cattle heat stress levels.

    Categories (adapted from Armstrong, 1994 & NRC standards):
    - THI < 68: Normal (Thermal comfort zone, no heat stress)
    - 68 <= THI < 72: Mild Heat Stress (Respiration rises, slight milk drop)
    - 72 <= THI < 79: Moderate Heat Stress (Significant drop in dry matter intake & fertility)
    - 79 <= THI < 84: Severe Heat Stress (Heavy panting, drooling, high reproductive failure)
    - THI >= 84: Danger / Very Severe (Emergency intervention required; heat stroke risk)

    Returns:
    - tuple: (Category Name, Severity Color Hex, Description / Action)
    """
    if thi is None:
        return "Unknown", "#888888", "Missing environmental sensor readings."

    if thi < 68.0:
        return (
            "Normal (Comfort Zone)",
            "#22c55e",  # Green
            "Optimal climatic conditions. Cows are in their thermal comfort zone."
        )
    elif thi < 72.0:
        return (
            "Mild Heat Stress",
            "#eab308",  # Yellow
            "Mild stress. Ensure adequate cool drinking water and ventilation."
        )
    elif thi < 79.0:
        return (
            "Moderate Heat Stress",
            "#f97316",  # Orange
            "Moderate stress. Activate barn fans, shade, and misters. Expect minor drop in milk yield."
        )
    elif thi < 84.0:
        return (
            "Severe Heat Stress",
            "#ef4444",  # Red
            "Severe stress. Substantial milk loss and fertility decline. Intensive cooling necessary."
        )
    else:
        return (
            "Emergency / Extreme Danger",
            "#7f1d1d",  # Dark Red
            "Extreme risk of heat stroke and mortality. Immediate emergency cooling and veterinary supervision required."
        )


def calculate_age(dob_str):
    """
    Calculates age in years and months from a Date of Birth string (YYYY-MM-DD).

    Parameters:
    - dob_str (str): Birthdate in "YYYY-MM-DD" format.

    Returns:
    - tuple: (years, months, formatted_string)
      Example: (3, 4, "3 yrs 4 mos")
    """
    if not dob_str:
        return 0, 0, "Unknown"

    try:
        birth_date = datetime.strptime(str(dob_str)[:10], "%Y-%m-%d").date()
        today = date.today()

        # Calculate difference
        years = today.year - birth_date.year
        months = today.month - birth_date.month

        if today.day < birth_date.day:
            months -= 1

        if months < 0:
            years -= 1
            months += 12

        years = max(0, years)
        months = max(0, months)

        formatted = f"{years} yrs {months} mos" if years > 0 else f"{months} mos"
        return years, months, formatted
    except Exception:
        return 0, 0, "Invalid Date"


def calculate_percentage_deviation(current_value, baseline_value):
    """
    Calculates the percentage change between a current observation and a baseline.

    Formula: ((current - baseline) / baseline) * 100

    Returns:
    - float: Percentage deviation (e.g. +45.2% or -28.1%), or 0.0 if baseline is zero.
    """
    if current_value is None or baseline_value is None:
        return 0.0
    try:
        cur = float(current_value)
        base = float(baseline_value)
        if base == 0.0:
            return 0.0
        pct = ((cur - base) / base) * 100.0
        return round(pct, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0
