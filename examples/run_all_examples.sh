#!/bin/bash

# Script to run all c2i2o examples

echo "========================================================================"
echo "Running all c2i2o examples"
echo "========================================================================"
echo ""

# Check if c2i2o is installed
if ! python -c "import c2i2o" 2>/dev/null; then
    echo "Error: c2i2o not installed"
    echo "Please run: pip install -e ."
    exit 1
fi

# Array of example scripts
examples=(
    "01_basic_parameters.py"
    "02_parameter_spaces.py"
    "03_simple_emulator.py"
    "04_inference_basics.py"
    "05_end_to_end_workflow.py"
)

# Run each example
for example in "${examples[@]}"; do
    echo "========================================================================"
    echo "Running: $example"
    echo "========================================================================"
    echo ""

    python "examples/$example"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "Error: $example failed with exit code $exit_code"
        exit $exit_code
    fi

    echo ""
    echo "✓ $example completed successfully"
    echo ""
done

echo "========================================================================"
echo "All examples completed successfully!"
echo "========================================================================"
echo ""
echo "Generated files:"
ls -lh examples/*.png 2>/dev/null || echo "  (No visualizations - install matplotlib)"
echo ""
