#!/usr/bin/env python3
"""
Edge TPU Model Converter

Converts TFLite models for Google Coral Edge TPU.

Usage:
    python3 convert_edgetpu.py --input model_int8.tflite --output model_edgetpu.tflite
"""

import argparse
import os
import subprocess
import sys


def check_compiler():
    """Check if edgetpu_compiler is available."""
    try:
        result = subprocess.run(
            ['edgetpu_compiler', '--version'],
            capture_output=True,
            text=True,
        )
        print(f'Edge TPU Compiler: {result.stdout.strip()}')
        return True
    except FileNotFoundError:
        print('Error: edgetpu_compiler not found', file=sys.stderr)
        print('Install from: https://coral.ai/docs/edgetpu/compiler/', file=sys.stderr)
        return False


def convert_model(input_path: str, output_dir: str = None):
    """Convert TFLite model for Edge TPU."""
    if not os.path.exists(input_path):
        print(f'Error: Input file not found: {input_path}', file=sys.stderr)
        return False

    if output_dir is None:
        output_dir = os.path.dirname(input_path) or '.'

    os.makedirs(output_dir, exist_ok=True)

    cmd = ['edgetpu_compiler', '-o', output_dir, input_path]

    print(f'Running: {" ".join(cmd)}')

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)

    if result.returncode != 0:
        print(f'Error: {result.stderr}', file=sys.stderr)
        return False

    # Find output file
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f'{base_name}_edgetpu.tflite')

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f'Output: {output_path} ({size_mb:.2f} MB)')
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description='Edge TPU Model Converter')
    parser.add_argument(
        '--input',
        '-i',
        required=True,
        help='Input TFLite model (must be int8 quantized)',
    )
    parser.add_argument(
        '--output-dir', '-o', help='Output directory (default: same as input)'
    )

    args = parser.parse_args()

    if not check_compiler():
        sys.exit(1)

    success = convert_model(args.input, args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
