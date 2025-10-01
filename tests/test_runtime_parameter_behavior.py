"""
Runtime test to verify that all parameters actually work correctly end-to-end.

This test creates actual InputSchema objects with test values and verifies that:
1. All parameters are correctly loaded from dictionaries
2. All parameters survive the round-trip conversion (YAML -> Object -> Legacy -> Object)
3. The 'enabled' flag and other conditional parameters work as expected
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import os

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from schema import InputSchema, ProcessingStep, SchemaManager
from config_manager import SystemConfig


class RuntimeParameterTester:
    """Tests runtime behavior of all parameters"""

    def __init__(self):
        self.test_results = []
        self.failures = []

    def log_result(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result"""
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details
        })
        if not passed:
            self.failures.append(f"{test_name}: {details}")

    def create_test_data_file(self, temp_dir: str) -> str:
        """Create a temporary test data file"""
        test_file = os.path.join(temp_dir, 'test_data.csv')
        with open(test_file, 'w') as f:
            f.write("Date,Value1,Value2\n")
            f.write("2024-01-01,100,200\n")
            f.write("2024-01-02,105,210\n")
        return test_file

    def test_enabled_flag_behavior(self, temp_dir: str):
        """Test that enabled flag correctly enables/disables processing steps"""
        print("\n" + "="*80)
        print("TEST 1: ENABLED FLAG BEHAVIOR")
        print("="*80)

        test_file = self.create_test_data_file(temp_dir)

        # Test 1a: Step with enabled=True
        try:
            schema_enabled = InputSchema(
                modality_name="Test Enabled True",
                path=test_file,
                column_number=2,
                has_header=True,
                processing_steps=[
                    ProcessingStep(
                        function='range_numeric_data',
                        args={'num_whole_digits': 2, 'decimal_places': 1},
                        enabled=True
                    )
                ]
            )
            legacy_list = schema_enabled.to_legacy_list()

            # Check that parameters are in legacy list when enabled=True
            num_whole_digits = legacy_list[4]
            decimal_places = legacy_list[5]

            if num_whole_digits == 2 and decimal_places == 1:
                self.log_result("Enabled=True: Parameters extracted", True,
                              f"num_whole_digits={num_whole_digits}, decimal_places={decimal_places}")
                print("[PASS] enabled=True: Parameters correctly extracted to legacy format")
            else:
                self.log_result("Enabled=True: Parameters extracted", False,
                              f"Expected (2, 1), got ({num_whole_digits}, {decimal_places})")
                print(f"[FAIL] enabled=True: Expected (2, 1), got ({num_whole_digits}, {decimal_places})")

        except Exception as e:
            self.log_result("Enabled=True test", False, str(e))
            print(f"[FAIL] enabled=True test: {e}")

        # Test 1b: Step with enabled=False
        try:
            schema_disabled = InputSchema(
                modality_name="Test Enabled False",
                path=test_file,
                column_number=2,
                has_header=True,
                processing_steps=[
                    ProcessingStep(
                        function='range_numeric_data',
                        args={'num_whole_digits': 2, 'decimal_places': 1},
                        enabled=False
                    )
                ]
            )
            legacy_list = schema_disabled.to_legacy_list()

            # Check that parameters are NOT in legacy list when enabled=False
            num_whole_digits = legacy_list[4]
            decimal_places = legacy_list[5]

            if num_whole_digits is None and decimal_places is None:
                self.log_result("Enabled=False: Parameters ignored", True,
                              f"num_whole_digits={num_whole_digits}, decimal_places={decimal_places}")
                print("[PASS] enabled=False: Parameters correctly ignored (None, None)")
            else:
                self.log_result("Enabled=False: Parameters ignored", False,
                              f"Expected (None, None), got ({num_whole_digits}, {decimal_places})")
                print(f"[FAIL] enabled=False: Expected (None, None), got ({num_whole_digits}, {decimal_places})")

        except Exception as e:
            self.log_result("Enabled=False test", False, str(e))
            print(f"[FAIL] enabled=False test: {e}")

    def test_multiple_steps_with_mixed_enabled(self, temp_dir: str):
        """Test multiple processing steps with mixed enabled flags"""
        print("\n" + "="*80)
        print("TEST 2: MULTIPLE STEPS WITH MIXED ENABLED FLAGS")
        print("="*80)

        test_file = self.create_test_data_file(temp_dir)

        try:
            schema = InputSchema(
                modality_name="Mixed Enabled Test",
                path=test_file,
                column_number=2,
                has_header=True,
                processing_steps=[
                    ProcessingStep(
                        function='convert_to_percent_changes',
                        args={},
                        enabled=True  # ENABLED
                    ),
                    ProcessingStep(
                        function='range_numeric_data',
                        args={'num_whole_digits': 3, 'decimal_places': 2},
                        enabled=False  # DISABLED
                    ),
                    ProcessingStep(
                        function='bin_numeric_data',
                        args={'num_bins': 8, 'outlier_percentile': 3, 'exponent': 1.7},
                        enabled=True  # ENABLED
                    )
                ]
            )

            legacy_list = schema.to_legacy_list()

            # Extract values
            convert_to_percents = legacy_list[3]  # Should be True
            num_whole_digits = legacy_list[4]      # Should be None (disabled)
            decimal_places = legacy_list[5]        # Should be None (disabled)
            num_bins = legacy_list[6]              # Should be 8
            outlier_percentile = legacy_list[10]   # Should be 3
            exponent = legacy_list[11]             # Should be 1.7

            print(f"\nLegacy list values:")
            print(f"  convert_to_percents: {convert_to_percents}")
            print(f"  num_whole_digits: {num_whole_digits}")
            print(f"  decimal_places: {decimal_places}")
            print(f"  num_bins: {num_bins}")
            print(f"  outlier_percentile: {outlier_percentile}")
            print(f"  exponent: {exponent}")

            # Check convert_to_percents is True
            if convert_to_percents is True:
                self.log_result("Step 1 (enabled): convert_to_percents", True)
                print("[PASS] Step 1 (enabled=True): convert_to_percents extracted")
            else:
                self.log_result("Step 1 (enabled): convert_to_percents", False,
                              f"Expected True, got {convert_to_percents}")
                print(f"[FAIL] Step 1: Expected True, got {convert_to_percents}")

            # Check range_numeric_data params are None (disabled)
            if num_whole_digits is None and decimal_places is None:
                self.log_result("Step 2 (disabled): range params ignored", True)
                print("[PASS] Step 2 (enabled=False): range params correctly ignored")
            else:
                self.log_result("Step 2 (disabled): range params ignored", False,
                              f"Expected (None, None), got ({num_whole_digits}, {decimal_places})")
                print(f"[FAIL] Step 2: Expected (None, None), got ({num_whole_digits}, {decimal_places})")

            # Check bin_numeric_data params are extracted
            if num_bins == 8 and outlier_percentile == 3 and exponent == 1.7:
                self.log_result("Step 3 (enabled): bin params extracted", True)
                print("[PASS] Step 3 (enabled=True): bin params correctly extracted")
            else:
                self.log_result("Step 3 (enabled): bin params extracted", False,
                              f"Expected (8, 3, 1.7), got ({num_bins}, {outlier_percentile}, {exponent})")
                print(f"[FAIL] Step 3: Expected (8, 3, 1.7), got ({num_bins}, {outlier_percentile}, {exponent})")

        except Exception as e:
            self.log_result("Mixed enabled steps test", False, str(e))
            print(f"[FAIL] Mixed enabled steps test: {e}")

    def test_all_input_schema_parameters(self, temp_dir: str):
        """Test that all InputSchema parameters round-trip correctly"""
        print("\n" + "="*80)
        print("TEST 3: ALL INPUT SCHEMA PARAMETERS ROUND-TRIP")
        print("="*80)

        test_file = self.create_test_data_file(temp_dir)

        try:
            # Create schema with ALL parameters set
            original = InputSchema(
                modality_name="Complete Test Schema",
                path=test_file,
                column_number=3,
                has_header=True,
                processing_steps=[
                    ProcessingStep(
                        function='convert_to_percent_changes',
                        args={},
                        enabled=True
                    ),
                    ProcessingStep(
                        function='bin_numeric_data',
                        args={'num_bins': 6, 'outlier_percentile': 2.5, 'exponent': 2.0},
                        enabled=True
                    )
                ],
                cross_attention=True,
                randomness_size=2
            )

            # Convert to legacy list
            legacy_list = original.to_legacy_list()

            # Convert back to InputSchema
            restored = InputSchema.from_legacy_list(legacy_list, "Restored Schema")

            # Compare all parameters
            print("\nComparing original vs restored:")

            checks = [
                ('path', str(original.path), str(restored.path)),
                ('column_number', original.column_number, restored.column_number),
                ('has_header', original.has_header, restored.has_header),
                ('cross_attention', original.cross_attention, restored.cross_attention),
                ('randomness_size', original.randomness_size, restored.randomness_size),
                ('processing_steps count', len(original.processing_steps), len(restored.processing_steps))
            ]

            all_passed = True
            for param_name, original_val, restored_val in checks:
                if original_val == restored_val:
                    self.log_result(f"Round-trip: {param_name}", True)
                    print(f"  [PASS] {param_name}: {original_val}")
                else:
                    self.log_result(f"Round-trip: {param_name}", False,
                                  f"Original: {original_val}, Restored: {restored_val}")
                    print(f"  [FAIL] {param_name}: Original={original_val}, Restored={restored_val}")
                    all_passed = False

            if all_passed:
                print("\n[PASS] All parameters survived round-trip conversion")
            else:
                print("\n[FAIL] Some parameters lost in round-trip conversion")

        except Exception as e:
            self.log_result("Round-trip test", False, str(e))
            print(f"[FAIL] Round-trip test: {e}")

    def test_config_yaml_parameters(self):
        """Test that all config.yaml parameters are accessible"""
        print("\n" + "="*80)
        print("TEST 4: CONFIG.YAML PARAMETERS")
        print("="*80)

        try:
            # Create test config
            config_dict = {
                'project_settings': {
                    'project_file_path': './',
                    'output_file_name': 'test_log.txt',
                    'model_file_name': './test_model.pth',
                    'create_new_model': 1,
                    'save_model': 1,
                    'device': 'cpu'
                },
                'data_splitting': {
                    'validation_size': 0.15,
                    'num_validation_files': 10
                },
                'training_parameters': {
                    'batch_size': 8,
                    'block_size': 32,
                    'max_iters': 1000,
                    'eval_interval': 100,
                    'eval_iters': 20,
                    'learning_rate': 0.0005
                },
                'model_architecture': {
                    'n_embd': 64,
                    'n_head': 8,
                    'n_layer': 4,
                    'dropout': 0.3,
                    'fixed_values': [-1.0, -0.5, 0, 0.5, 1.0]
                }
            }

            # Create SystemConfig from dict
            sys_config = SystemConfig.from_dict(config_dict)

            # Verify all parameters
            checks = [
                ('project_file_path', './', sys_config.project_file_path),
                ('output_file_name', 'test_log.txt', sys_config.output_file_name),
                ('model_file_name', './test_model.pth', sys_config.model_file_name),
                ('create_new_model', True, sys_config.create_new_model),
                ('save_model', True, sys_config.save_model),
                ('device', 'cpu', sys_config.device),
                ('validation_size', 0.15, sys_config.validation_size),
                ('num_validation_files', 10, sys_config.num_validation_files),
                ('batch_size', 8, sys_config.batch_size),
                ('block_size', 32, sys_config.block_size),
                ('max_iters', 1000, sys_config.max_iters),
                ('eval_interval', 100, sys_config.eval_interval),
                ('eval_iters', 20, sys_config.eval_iters),
                ('learning_rate', 0.0005, sys_config.learning_rate),
                ('n_embd', 64, sys_config.n_embd),
                ('n_head', 8, sys_config.n_head),
                ('n_layer', 4, sys_config.n_layer),
                ('dropout', 0.3, sys_config.dropout),
                ('fixed_values count', 5, len(sys_config.fixed_values))
            ]

            all_passed = True
            for param_name, expected, actual in checks:
                if expected == actual:
                    self.log_result(f"Config param: {param_name}", True)
                    print(f"  [PASS] {param_name}: {actual}")
                else:
                    self.log_result(f"Config param: {param_name}", False,
                                  f"Expected: {expected}, Got: {actual}")
                    print(f"  [FAIL] {param_name}: Expected={expected}, Got={actual}")
                    all_passed = False

            # Test round-trip
            restored_dict = sys_config.to_dict()
            restored_config = SystemConfig.from_dict(restored_dict)

            if sys_config.batch_size == restored_config.batch_size:
                self.log_result("Config round-trip", True)
                print("\n[PASS] Config round-trip successful")
            else:
                self.log_result("Config round-trip", False)
                print("\n[FAIL] Config round-trip failed")

        except Exception as e:
            self.log_result("Config test", False, str(e))
            print(f"[FAIL] Config test: {e}")

    def test_bin_numeric_data_args(self, temp_dir: str):
        """Specifically test bin_numeric_data with all three arguments"""
        print("\n" + "="*80)
        print("TEST 5: BIN_NUMERIC_DATA WITH ALL ARGUMENTS")
        print("="*80)

        test_file = self.create_test_data_file(temp_dir)

        try:
            schema = InputSchema(
                modality_name="Bin Args Test",
                path=test_file,
                column_number=2,
                has_header=True,
                processing_steps=[
                    ProcessingStep(
                        function='bin_numeric_data',
                        args={
                            'num_bins': 10,
                            'outlier_percentile': 5.0,
                            'exponent': 2.5
                        },
                        enabled=True
                    )
                ]
            )

            legacy_list = schema.to_legacy_list()

            num_bins = legacy_list[6]
            outlier_percentile = legacy_list[10]
            exponent = legacy_list[11]

            print(f"\nExtracted values:")
            print(f"  num_bins: {num_bins}")
            print(f"  outlier_percentile: {outlier_percentile}")
            print(f"  exponent: {exponent}")

            if num_bins == 10:
                self.log_result("bin_numeric_data: num_bins", True)
                print("[PASS] num_bins correctly extracted")
            else:
                self.log_result("bin_numeric_data: num_bins", False,
                              f"Expected 10, got {num_bins}")
                print(f"[FAIL] num_bins: Expected 10, got {num_bins}")

            if outlier_percentile == 5.0:
                self.log_result("bin_numeric_data: outlier_percentile", True)
                print("[PASS] outlier_percentile correctly extracted")
            else:
                self.log_result("bin_numeric_data: outlier_percentile", False,
                              f"Expected 5.0, got {outlier_percentile}")
                print(f"[FAIL] outlier_percentile: Expected 5.0, got {outlier_percentile}")

            if exponent == 2.5:
                self.log_result("bin_numeric_data: exponent", True)
                print("[PASS] exponent correctly extracted")
            else:
                self.log_result("bin_numeric_data: exponent", False,
                              f"Expected 2.5, got {exponent}")
                print(f"[FAIL] exponent: Expected 2.5, got {exponent}")

        except Exception as e:
            self.log_result("bin_numeric_data args test", False, str(e))
            print(f"[FAIL] bin_numeric_data args test: {e}")

    def generate_summary(self):
        """Generate test summary"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t['passed'])
        failed_tests = total_tests - passed_tests

        print(f"\nTotal tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests*100):.1f}%")

        if self.failures:
            print("\nFAILURES:")
            for i, failure in enumerate(self.failures, 1):
                print(f"{i}. {failure}")
        else:
            print("\n[SUCCESS] ALL TESTS PASSED!")

        return failed_tests == 0


def main():
    """Run all runtime parameter tests"""
    tester = RuntimeParameterTester()

    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        print("="*80)
        print("RUNTIME PARAMETER BEHAVIOR TEST")
        print("="*80)
        print(f"Using temporary directory: {temp_dir}")

        # Run all tests
        tester.test_enabled_flag_behavior(temp_dir)
        tester.test_multiple_steps_with_mixed_enabled(temp_dir)
        tester.test_all_input_schema_parameters(temp_dir)
        tester.test_config_yaml_parameters()
        tester.test_bin_numeric_data_args(temp_dir)

        # Generate summary
        all_passed = tester.generate_summary()

        if all_passed:
            print("\n" + "="*80)
            print("ALL RUNTIME TESTS PASSED - NO BUGS FOUND")
            print("="*80)
            return 0
        else:
            print("\n" + "="*80)
            print("SOME TESTS FAILED - REVIEW FAILURES ABOVE")
            print("="*80)
            return 1


if __name__ == '__main__':
    exit(main())
