"""
Comprehensive test to identify all parameters that might be ignored during YAML to legacy conversion.

This test traces every parameter from both input_schemas.yaml and config.yaml through the
conversion process to identify bugs similar to the 'enabled' flag issue.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Set
from dataclasses import dataclass


@dataclass
class ParameterTest:
    """Test result for a single parameter"""
    parameter_name: str
    location: str  # YAML path like "modalities[0].processing_steps[0].enabled"
    found_in_yaml: bool = False
    extracted_in_code: bool = False
    checked_for_conditions: bool = False
    converted_to_legacy: bool = False
    issues: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class ParameterAnalyzer:
    """Analyzes all parameters in YAML files and their conversion"""

    def __init__(self):
        self.input_schema_params: List[ParameterTest] = []
        self.config_params: List[ParameterTest] = []
        self.all_issues: List[str] = []

    def analyze_input_schemas(self, yaml_path: str):
        """Analyze all parameters in input_schemas.yaml"""
        print("\n" + "="*80)
        print("ANALYZING INPUT_SCHEMAS.YAML")
        print("="*80)

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        modalities = data.get('modalities', [])

        for mod_idx, modality in enumerate(modalities):
            mod_path = f"modalities[{mod_idx}]"

            # Top-level modality parameters
            for key in modality.keys():
                param = ParameterTest(
                    parameter_name=key,
                    location=f"{mod_path}.{key}",
                    found_in_yaml=True
                )
                self.input_schema_params.append(param)

                # Special handling for processing_steps
                if key == 'processing_steps':
                    steps = modality.get('processing_steps', [])
                    for step_idx, step in enumerate(steps):
                        step_path = f"{mod_path}.processing_steps[{step_idx}]"
                        for step_key in step.keys():
                            step_param = ParameterTest(
                                parameter_name=f"processing_steps.{step_key}",
                                location=f"{step_path}.{step_key}",
                                found_in_yaml=True
                            )
                            self.input_schema_params.append(step_param)

                            # Deeper analysis for args
                            if step_key == 'args':
                                args = step.get('args', {})
                                for arg_key in args.keys():
                                    arg_param = ParameterTest(
                                        parameter_name=f"processing_steps.args.{arg_key}",
                                        location=f"{step_path}.args.{arg_key}",
                                        found_in_yaml=True
                                    )
                                    self.input_schema_params.append(arg_param)

        print(f"\nTotal parameters found in input_schemas.yaml: {len(self.input_schema_params)}")

    def analyze_config(self, yaml_path: str):
        """Analyze all parameters in config.yaml"""
        print("\n" + "="*80)
        print("ANALYZING CONFIG.YAML")
        print("="*80)

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Flatten nested structure
        for section_name, section_data in data.items():
            if isinstance(section_data, dict):
                for key in section_data.keys():
                    param = ParameterTest(
                        parameter_name=key,
                        location=f"{section_name}.{key}",
                        found_in_yaml=True
                    )
                    self.config_params.append(param)

        print(f"\nTotal parameters found in config.yaml: {len(self.config_params)}")

    def trace_input_schema_conversion(self):
        """Trace how input_schemas.yaml parameters are converted to legacy format"""
        print("\n" + "="*80)
        print("TRACING INPUT_SCHEMAS.YAML CONVERSION")
        print("="*80)

        # Read schema.py to analyze conversion logic
        schema_path = Path(__file__).parent / 'schema.py'
        with open(schema_path, 'r') as f:
            schema_code = f.read()

        # Check each parameter
        for param in self.input_schema_params:
            param_name = param.parameter_name.split('.')[-1]

            # Check if parameter is extracted in from_dict
            if f"config_dict.get('{param_name}'" in schema_code or f"config_dict['{param_name}']" in schema_code:
                param.extracted_in_code = True
            elif f"step_dict.get('{param_name}'" in schema_code or f"**step_dict" in schema_code:
                param.extracted_in_code = True

            # Check if parameter is used in to_legacy_list
            if param_name in schema_code and 'to_legacy_list' in schema_code:
                # Look for conditional checks
                lines = schema_code.split('\n')
                for i, line in enumerate(lines):
                    if param_name in line and 'to_legacy_list' in '\n'.join(lines[max(0, i-50):i+10]):
                        if 'if' in line or 'for' in line:
                            param.checked_for_conditions = True
                        if 'legacy_list' in '\n'.join(lines[i:min(len(lines), i+5)]):
                            param.converted_to_legacy = True

        # Identify issues
        critical_params = ['enabled', 'modality_name', 'path', 'column_number', 'has_header',
                          'cross_attention', 'randomness_size', 'function', 'args']

        for param in self.input_schema_params:
            base_name = param.parameter_name.split('.')[-1]

            if param.found_in_yaml and not param.extracted_in_code:
                issue = f"YAML parameter '{param.location}' is NOT extracted in schema.py"
                param.issues.append(issue)
                self.all_issues.append(issue)

            if base_name in critical_params and param.extracted_in_code and not param.checked_for_conditions:
                # Special case for 'enabled' which we know should be checked
                if base_name == 'enabled':
                    issue = f"CRITICAL: '{param.location}' extracted but condition check may be missing"
                    param.issues.append(issue)
                    self.all_issues.append(issue)

    def trace_config_conversion(self):
        """Trace how config.yaml parameters are converted"""
        print("\n" + "="*80)
        print("TRACING CONFIG.YAML CONVERSION")
        print("="*80)

        # Read config_manager.py to analyze conversion logic
        config_path = Path(__file__).parent / 'config_manager.py'
        with open(config_path, 'r') as f:
            config_code = f.read()

        # Read compatibility_layer.py for get_system_parameters
        compat_path = Path(__file__).parent / 'compatibility_layer.py'
        with open(compat_path, 'r') as f:
            compat_code = f.read()

        for param in self.config_params:
            param_name = param.parameter_name

            # Check if in SystemConfig dataclass
            if f"{param_name}:" in config_code:
                param.extracted_in_code = True

            # Check if in from_dict method
            if f"'{param_name}'" in config_code:
                param.checked_for_conditions = True

            # Check if returned in get_system_parameters
            if f"'{param_name}':" in compat_code and 'get_system_parameters' in compat_code:
                param.converted_to_legacy = True

        # Identify issues
        for param in self.config_params:
            if param.found_in_yaml and not param.extracted_in_code:
                issue = f"CONFIG parameter '{param.location}' is NOT extracted in config_manager.py"
                param.issues.append(issue)
                self.all_issues.append(issue)

            if param.extracted_in_code and not param.converted_to_legacy:
                issue = f"CONFIG parameter '{param.location}' extracted but NOT returned in get_system_parameters()"
                param.issues.append(issue)
                self.all_issues.append(issue)

    def check_enabled_flag_specifically(self):
        """Deep dive into the enabled flag bug"""
        print("\n" + "="*80)
        print("DEEP ANALYSIS: ENABLED FLAG HANDLING")
        print("="*80)

        schema_path = Path(__file__).parent / 'schema.py'
        with open(schema_path, 'r') as f:
            lines = f.readlines()

        # Find to_legacy_list method
        in_to_legacy = False
        for_loop_found = False
        enabled_check_found = False

        for line_num, line in enumerate(lines, 1):
            if 'def to_legacy_list(self)' in line:
                in_to_legacy = True
                print(f"\nLine {line_num}: Found to_legacy_list() method")

            if in_to_legacy and 'for step in self.processing_steps' in line:
                for_loop_found = True
                print(f"Line {line_num}: Found loop over processing_steps")

            if in_to_legacy and for_loop_found and 'step.enabled' in line:
                enabled_check_found = True
                print(f"Line {line_num}: ENABLED CHECK FOUND: {line.strip()}")

            if in_to_legacy and line.strip().startswith('def ') and 'to_legacy_list' not in line:
                break

        if for_loop_found and not enabled_check_found:
            issue = "BUG CONFIRMED: to_legacy_list() loops through processing_steps but doesn't check step.enabled"
            self.all_issues.append(issue)
            print(f"\n{'!'*80}")
            print(issue)
            print(f"{'!'*80}")
        elif enabled_check_found:
            print(f"\nGOOD: enabled flag is being checked in to_legacy_list()")

    def check_missing_parameters(self):
        """Check for parameters that exist in one place but not another"""
        print("\n" + "="*80)
        print("CHECKING FOR MISSING PARAMETERS")
        print("="*80)

        # Check if all bin_numeric_data args are extracted
        bin_args = ['num_bins', 'outlier_percentile', 'exponent']
        schema_path = Path(__file__).parent / 'schema.py'
        with open(schema_path, 'r') as f:
            schema_code = f.read()

        print("\nChecking bin_numeric_data arguments:")
        for arg in bin_args:
            if f"{arg} = step.args.get('{arg}')" in schema_code:
                print(f"  [OK] {arg}: extracted in to_legacy_list()")
            else:
                issue = f"POTENTIAL BUG: bin_numeric_data arg '{arg}' may not be extracted in to_legacy_list()"
                print(f"  [MISSING] {arg}: {issue}")
                self.all_issues.append(issue)

        # Check if outlier_percentile and exponent are in legacy list
        if 'outlier_percentile,' in schema_code and 'exponent' in schema_code:
            print("\n[OK] outlier_percentile and exponent are added to legacy_list")
        else:
            print("\n[MISSING] outlier_percentile and exponent may not be in legacy_list")

    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE PARAMETER TEST REPORT")
        print("="*80)

        # Summary statistics
        print("\n1. INPUT_SCHEMAS.YAML PARAMETERS")
        print("-" * 80)
        total_input = len(self.input_schema_params)
        extracted_input = sum(1 for p in self.input_schema_params if p.extracted_in_code)
        converted_input = sum(1 for p in self.input_schema_params if p.converted_to_legacy)
        issues_input = sum(1 for p in self.input_schema_params if p.issues)

        print(f"Total parameters: {total_input}")
        print(f"Extracted in code: {extracted_input}/{total_input}")
        print(f"Converted to legacy: {converted_input}/{total_input}")
        print(f"Parameters with issues: {issues_input}/{total_input}")

        print("\n2. CONFIG.YAML PARAMETERS")
        print("-" * 80)
        total_config = len(self.config_params)
        extracted_config = sum(1 for p in self.config_params if p.extracted_in_code)
        converted_config = sum(1 for p in self.config_params if p.converted_to_legacy)
        issues_config = sum(1 for p in self.config_params if p.issues)

        print(f"Total parameters: {total_config}")
        print(f"Extracted in code: {extracted_config}/{total_config}")
        print(f"Converted to legacy: {converted_config}/{total_config}")
        print(f"Parameters with issues: {issues_config}/{total_config}")

        # List all parameters
        print("\n3. DETAILED PARAMETER LIST - INPUT_SCHEMAS.YAML")
        print("-" * 80)
        unique_params = {}
        for param in self.input_schema_params:
            base_name = param.parameter_name
            if base_name not in unique_params:
                unique_params[base_name] = param

        for param_name in sorted(unique_params.keys()):
            param = unique_params[param_name]
            status = "[OK]" if param.extracted_in_code else "[MISS]"
            print(f"{status} {param.parameter_name:<40} | Extracted: {param.extracted_in_code} | Converted: {param.converted_to_legacy}")

        print("\n4. DETAILED PARAMETER LIST - CONFIG.YAML")
        print("-" * 80)
        for param in sorted(self.config_params, key=lambda p: p.location):
            status = "[OK]" if param.extracted_in_code and param.converted_to_legacy else "[MISS]"
            print(f"{status} {param.parameter_name:<30} | Extracted: {param.extracted_in_code} | Converted: {param.converted_to_legacy}")

        # Issues
        print("\n5. ALL ISSUES FOUND")
        print("-" * 80)
        if not self.all_issues:
            print("[OK] NO ISSUES FOUND - All parameters are properly handled!")
        else:
            for i, issue in enumerate(self.all_issues, 1):
                print(f"{i}. {issue}")

        # Critical findings
        print("\n6. CRITICAL FINDINGS")
        print("-" * 80)

        critical_issues = [issue for issue in self.all_issues if 'CRITICAL' in issue or 'BUG' in issue]
        if critical_issues:
            print(f"Found {len(critical_issues)} critical issues:")
            for issue in critical_issues:
                print(f"  ! {issue}")
        else:
            print("No critical issues found.")

        # Specific parameter checks
        print("\n7. SPECIFIC PARAMETER STATUS")
        print("-" * 80)

        critical_params = {
            'enabled': 'Controls whether processing steps are active',
            'modality_name': 'Identifies the modality',
            'path': 'Data file path',
            'column_number': 'Column to extract from data',
            'has_header': 'Whether CSV has headers',
            'cross_attention': 'Enable cross-attention mechanism',
            'randomness_size': 'Add randomness to data',
            'function': 'Processing function name',
            'args': 'Processing function arguments',
            'num_bins': 'Number of bins for binning',
            'outlier_percentile': 'Outlier threshold for binning',
            'exponent': 'Bin distribution exponent'
        }

        for param_name, description in critical_params.items():
            matching = [p for p in self.input_schema_params if param_name in p.parameter_name]
            if matching:
                param = matching[0]
                status = "[OK]" if param.extracted_in_code else "[MISS]"
                print(f"{status} {param_name:<25} - {description}")
                if param.issues:
                    for issue in param.issues:
                        print(f"    -> ISSUE: {issue}")
            else:
                print(f"[?] {param_name:<25} - {description} (NOT FOUND IN YAML)")


def main():
    """Run comprehensive parameter analysis"""
    analyzer = ParameterAnalyzer()

    # Paths
    input_schema_path = Path(__file__).parent / 'input_schemas.yaml'
    config_path = Path(__file__).parent / 'config.yaml'

    # Analyze YAML files
    analyzer.analyze_input_schemas(str(input_schema_path))
    analyzer.analyze_config(str(config_path))

    # Trace conversion
    analyzer.trace_input_schema_conversion()
    analyzer.trace_config_conversion()

    # Specific checks
    analyzer.check_enabled_flag_specifically()
    analyzer.check_missing_parameters()

    # Generate final report
    analyzer.generate_report()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
