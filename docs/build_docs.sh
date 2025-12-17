#!/bin/bash
# Script to build documentation

set -e

echo "Building c2i2o documentation..."
echo ""

# Change to docs directory
cd "$(dirname "$0")"

# Clean previous builds
echo "Cleaning previous builds..."
make clean

# Build HTML documentation
echo "Building HTML documentation..."
make html

echo ""
echo "Documentation built successfully!"
echo "Open: $(pwd)/build/html/index.html"
