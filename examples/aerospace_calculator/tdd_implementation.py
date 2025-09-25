#!/usr/bin/env python3
"""
AEROSPACE CALCULATOR - TDD IMPLEMENTATION EXAMPLE
Demonstrates pure TDD-first development for aerospace engineering
"""

import math
import unittest
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class OrbitalElements:
    """Orbital elements for satellite/spacecraft"""
    semi_major_axis: float  # meters
    eccentricity: float     # dimensionless
    inclination: float      # degrees
    raan: float            # degrees
    arg_perigee: float     # degrees
    true_anomaly: float    # degrees

@dataclass
class StateVector:
    """Position and velocity vectors"""
    position: Tuple[float, float, float]  # meters
    velocity: Tuple[float, float, float]  # m/s

class AerospaceCalculatorResult:
    """Result structure for aerospace calculations"""

    def __init__(self, success: bool, data: Any, message: str, metadata: Dict[str, Any] = None):
        self.success = success
        self.data = data
        self.message = message
        self.metadata = metadata or {}

class AerospaceCalculator:
    """
    Advanced aerospace calculator with real physics calculations
    Built using pure Test-Driven Development methodology
    """

    # Physical constants
    EARTH_MU = 3.986004418e14  # Earth's gravitational parameter [m³/s²]
    EARTH_RADIUS = 6371000.0   # Earth radius [m]
    SUN_MU = 1.3271244e20      # Sun's gravitational parameter [m³/s²]

    def __init__(self):
        """Initialize the aerospace calculator"""
        self.logger = None  # In real implementation, would have logging
        self.config = self._load_configuration()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration settings"""
        return {
            'precision': 1e-10,
            'max_iterations': 1000,
            'timeout': 30.0
        }

    def validate_input(self, input_data: Any) -> AerospaceCalculatorResult:
        """
        Validate input parameters for aerospace calculations
        """
        if input_data is None:
            return AerospaceCalculatorResult(
                success=False,
                data=None,
                message="Input data cannot be None",
                metadata={'error_type': 'null_input'}
            )

        if not isinstance(input_data, (int, float, dict, list)):
            return AerospaceCalculatorResult(
                success=False,
                data=None,
                message="Input must be numeric, dict, or list",
                metadata={'error_type': 'invalid_type'}
            )

        return AerospaceCalculatorResult(
            success=True,
            data=input_data,
            message="Input validation passed"
        )

    def calculate_orbital_velocity(self, altitude: float) -> AerospaceCalculatorResult:
        """
        Calculate orbital velocity for circular orbit
        """
        try:
            validation = self.validate_input(altitude)
            if not validation.success:
                return validation

            if altitude < 0:
                return AerospaceCalculatorResult(
                    success=False,
                    data=None,
                    message="Altitude cannot be negative",
                    metadata={'error_type': 'invalid_altitude'}
                )

            radius = self.EARTH_RADIUS + altitude
            velocity = math.sqrt(self.EARTH_MU / radius)

            return AerospaceCalculatorResult(
                success=True,
                data=velocity,
                message="Orbital velocity calculated successfully",
                metadata={
                    'altitude': altitude,
                    'radius': radius,
                    'calculation_method': 'circular_orbit'
                }
            )

        except Exception as e:
            return AerospaceCalculatorResult(
                success=False,
                data=None,
                message=f"Error calculating orbital velocity: {str(e)}",
                metadata={'error_type': 'calculation_error'}
            )

    def calculate_escape_velocity(self, altitude: float) -> AerospaceCalculatorResult:
        """
        Calculate escape velocity from Earth's gravity
        """
        try:
            validation = self.validate_input(altitude)
            if not validation.success:
                return validation

            radius = self.EARTH_RADIUS + altitude
            escape_velocity = math.sqrt(2 * self.EARTH_MU / radius)

            return AerospaceCalculatorResult(
                success=True,
                data=escape_velocity,
                message="Escape velocity calculated successfully",
                metadata={
                    'altitude': altitude,
                    'radius': radius,
                    'calculation_method': 'escape_trajectory'
                }
            )

        except Exception as e:
            return AerospaceCalculatorResult(
                success=False,
                data=None,
                message=f"Error calculating escape velocity: {str(e)}",
                metadata={'error_type': 'calculation_error'}
            )

    def calculate_hohmann_transfer(self, r1: float, r2: float) -> AerospaceCalculatorResult:
        """
        Calculate Hohmann transfer orbit parameters
        """
        try:
            # Validate inputs
            for radius in [r1, r2]:
                validation = self.validate_input(radius)
                if not validation.success:
                    return validation

            if r1 <= self.EARTH_RADIUS or r2 <= self.EARTH_RADIUS:
                return AerospaceCalculatorResult(
                    success=False,
                    data=None,
                    message="Radii must be greater than Earth radius",
                    metadata={'error_type': 'invalid_orbit'}
                )

            # Calculate transfer parameters
            a_transfer = (r1 + r2) / 2

            v1 = math.sqrt(self.EARTH_MU / r1)
            v2 = math.sqrt(self.EARTH_MU / r2)
            vt1 = math.sqrt(self.EARTH_MU * (2/r1 - 1/a_transfer))
            vt2 = math.sqrt(self.EARTH_MU * (2/r2 - 1/a_transfer))

            delta_v1 = vt1 - v1
            delta_v2 = v2 - vt2
            total_delta_v = delta_v1 + delta_v2
            transfer_time = math.pi * math.sqrt(a_transfer**3 / self.EARTH_MU)

            result_data = {
                'transfer_sma': a_transfer,
                'initial_velocity': v1,
                'final_velocity': v2,
                'transfer_velocity_start': vt1,
                'transfer_velocity_end': vt2,
                'delta_v_start': delta_v1,
                'delta_v_end': delta_v2,
                'total_delta_v': total_delta_v,
                'transfer_time': transfer_time,  # seconds
                'transfer_time_hours': transfer_time / 3600
            }

            return AerospaceCalculatorResult(
                success=True,
                data=result_data,
                message="Hohmann transfer calculated successfully",
                metadata={
                    'initial_radius': r1,
                    'final_radius': r2,
                    'calculation_method': 'hohmann_transfer'
                }
            )

        except Exception as e:
            return AerospaceCalculatorResult(
                success=False,
                data=None,
                message=f"Error calculating Hohmann transfer: {str(e)}",
                metadata={'error_type': 'calculation_error'}
            )

    def calculate_launch_azimuth(self, launch_lat: float, target_inclination: float) -> AerospaceCalculatorResult:
        """
        Calculate optimal launch azimuth for given latitude and inclination
        """
        try:
            # Validate inputs
            for value in [launch_lat, target_inclination]:
                validation = self.validate_input(value)
                if not validation.success:
                    return validation

            # Clamp values to valid range
            launch_lat = max(-90, min(90, launch_lat))
            target_inclination = max(0, min(180, target_inclination))

            launch_lat_rad = math.radians(launch_lat)
            target_inc_rad = math.radians(target_inclination)

            cos_inc = math.cos(target_inc_rad)
            cos_lat = math.cos(launch_lat_rad)

            if cos_lat == 0:
                azimuth = 90.0 if launch_lat > 0 else -90.0
            else:
                argument = cos_inc / cos_lat
                argument = max(-1.0, min(1.0, argument))  # Clamp to [-1, 1]
                azimuth = math.degrees(math.asin(argument))

            result_data = {
                'azimuth': azimuth,
                'launch_latitude': launch_lat,
                'target_inclination': target_inclination,
                'azimuth_range': [azimuth - 5, azimuth + 5]  # ±5° window
            }

            return AerospaceCalculatorResult(
                success=True,
                data=result_data,
                message="Launch azimuth calculated successfully",
                metadata={
                    'calculation_method': 'launch_azimuth_optimization',
                    'latitude_range': [-90, 90],
                    'inclination_range': [0, 180]
                }
            )

        except Exception as e:
            return AerospaceCalculatorResult(
                success=False,
                data=None,
                message=f"Error calculating launch azimuth: {str(e)}",
                metadata={'error_type': 'calculation_error'}
            )

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health information"""
        return {
            'status': 'operational',
            'domain': 'aerospace',
            'version': '2.0.0',
            'supported_calculations': [
                'orbital_velocity',
                'escape_velocity',
                'hohmann_transfer',
                'launch_azimuth'
            ],
            'constants': {
                'earth_mu': self.EARTH_MU,
                'earth_radius': self.EARTH_RADIUS,
                'sun_mu': self.SUN_MU
            }
        }

# Factory function for easy instantiation
def create_aerospace_calculator() -> AerospaceCalculator:
    """Factory function to create aerospace calculator instance"""
    return AerospaceCalculator()

# Test suite demonstrating TDD approach
class TestAerospaceCalculator(unittest.TestCase):
    """Comprehensive test suite for aerospace calculator"""

    def setUp(self):
        """Set up test fixtures"""
        self.calculator = create_aerospace_calculator()

    def tearDown(self):
        """Clean up after tests"""
        pass

    # Unit Tests
    def test_orbital_velocity_calculation(self):
        """Test orbital velocity calculation"""
        # Test LEO
        result = self.calculator.calculate_orbital_velocity(400000)  # 400km altitude
        self.assertTrue(result.success)
        self.assertGreater(result.data, 7600)  # Should be around 7672 m/s
        self.assertLess(result.data, 7800)

        # Test GEO
        result = self.calculator.calculate_orbital_velocity(35786000)  # GEO altitude
        self.assertTrue(result.success)
        self.assertGreater(result.data, 3000)
        self.assertLess(result.data, 3200)

    def test_escape_velocity_calculation(self):
        """Test escape velocity calculation"""
        result = self.calculator.calculate_escape_velocity(400000)
        self.assertTrue(result.success)
        self.assertGreater(result.data, 10800)  # Should be around 10850 m/s
        self.assertLess(result.data, 11200)

    def test_hohmann_transfer_calculation(self):
        """Test Hohmann transfer calculation"""
        r1 = 6371000 + 400000  # LEO
        r2 = 6371000 + 35786000  # GEO

        result = self.calculator.calculate_hohmann_transfer(r1, r2)
        self.assertTrue(result.success)
        self.assertIn('total_delta_v', result.data)
        self.assertIn('transfer_time_hours', result.data)
        self.assertGreater(result.data['total_delta_v'], 3800)
        self.assertLess(result.data['total_delta_v'], 4200)

    def test_launch_azimuth_calculation(self):
        """Test launch azimuth calculation"""
        result = self.calculator.calculate_launch_azimuth(28.5, 51.6)  # KSC to ISS
        self.assertTrue(result.success)
        self.assertIn('azimuth', result.data)
        self.assertIn('azimuth_range', result.data)

    # Edge Case Tests
    def test_negative_altitude(self):
        """Test handling of negative altitude"""
        result = self.calculator.calculate_orbital_velocity(-1000)
        self.assertFalse(result.success)
        self.assertIn('negative', result.message.lower())

    def test_zero_altitude(self):
        """Test handling of zero altitude"""
        result = self.calculator.calculate_orbital_velocity(0)
        self.assertTrue(result.success)
        self.assertGreater(result.data, 7800)  # Surface orbital velocity

    def test_invalid_input_types(self):
        """Test handling of invalid input types"""
        test_cases = [None, "invalid", [], {}]

        for invalid_input in test_cases:
            result = self.calculator.validate_input(invalid_input)
            self.assertFalse(result.success)

    # Integration Tests
    def test_end_to_end_trajectory_calculation(self):
        """Test complete trajectory calculation workflow"""
        # Calculate orbital velocity
        leo_result = self.calculator.calculate_orbital_velocity(400000)
        self.assertTrue(leo_result.success)

        # Calculate Hohmann transfer to GEO
        geo_alt = 35786000
        r_leo = 6371000 + 400000
        r_geo = 6371000 + geo_alt

        transfer_result = self.calculator.calculate_hohmann_transfer(r_leo, r_geo)
        self.assertTrue(transfer_result.success)

        # Verify reasonable transfer time
        self.assertGreater(transfer_result.data['transfer_time_hours'], 5)
        self.assertLess(transfer_result.data['transfer_time_hours'], 20)

    def test_multiple_calculations_consistency(self):
        """Test consistency across multiple calculations"""
        altitude = 500000  # 500km altitude

        # Get orbital and escape velocities
        orbital_result = self.calculator.calculate_orbital_velocity(altitude)
        escape_result = self.calculator.calculate_escape_velocity(altitude)

        self.assertTrue(orbital_result.success)
        self.assertTrue(escape_result.success)

        # Escape velocity should be sqrt(2) times orbital velocity
        expected_escape = orbital_result.data * math.sqrt(2)
        self.assertAlmostEqual(escape_result.data, expected_escape, places=2)

    # Performance Tests
    def test_calculation_performance(self):
        """Test calculation performance"""
        import time

        altitudes = [alt for alt in range(0, 1000000, 100000)]  # 0 to 1000km

        start_time = time.time()
        for altitude in altitudes:
            result = self.calculator.calculate_orbital_velocity(altitude)
            self.assertTrue(result.success)
        end_time = time.time()

        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 1.0)  # Less than 1 second

if __name__ == "__main__":
    # Run comprehensive test suite
    unittest.main(verbosity=2)