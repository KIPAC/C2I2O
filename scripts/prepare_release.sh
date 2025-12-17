#!/bin/bash
# Script to prepare for a new release

set -e

echo "=========================================="
echo "c2i2o Release Preparation"
echo "=========================================="
echo ""

# Get version
VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.0"
    exit 1
fi

echo "Preparing release v$VERSION"
echo ""

# Check we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "Error: Must be on main branch (currently on $BRANCH)"
    exit 1
fi

# Check working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean"
    git status --short
    exit 1
fi

echo "✓ On main branch with clean working directory"
echo ""

# Run tests
echo "Running tests..."
pytest --quiet
echo "✓ Tests passed"
echo ""

# Run pre-commit
echo "Running pre-commit checks..."
pre-commit run --all-files
echo "✓ Pre-commit checks passed"
echo ""

# Build docs
echo "Building documentation..."
cd docs
make clean > /dev/null 2>&1
make html > /dev/null 2>&1
cd ..
echo "✓ Documentation built"
echo ""

# Run examples
echo "Running examples..."
bash examples/run_all_examples.sh > /dev/null 2>&1
echo "✓ Examples ran successfully"
echo ""

# Update version in pyproject.toml
echo "Updating version in pyproject.toml..."
sed -i.bak "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
rm pyproject.toml.bak
echo "✓ Updated pyproject.toml"
echo ""

# Update version in docs
echo "Updating version in docs/source/conf.py..."
sed -i.bak "s/^release = .*/release = '$VERSION'/" docs/source/conf.py
rm docs/source/conf.py.bak
echo "✓ Updated docs/source/conf.py"
echo ""

# Check CHANGELOG
echo "Checking CHANGELOG.md..."
if ! grep -q "\[$VERSION\]" CHANGELOG.md; then
    echo "Warning: CHANGELOG.md doesn't mention version $VERSION"
    echo "Please update CHANGELOG.md before proceeding"
    exit 1
fi
echo "✓ CHANGELOG.md mentions v$VERSION"
echo ""

# Build package
echo "Building package..."
rm -rf dist/ build/ src/*.egg-info
python -m build > /dev/null 2>&1
echo "✓ Package built"
echo ""

# Check package
echo "Checking package..."
twine check dist/* > /dev/null 2>&1
echo "✓ Package checks passed"
echo ""

# Summary
echo "=========================================="
echo "Release v$VERSION is ready!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review changes:"
echo "     git diff pyproject.toml docs/source/conf.py"
echo ""
echo "  2. Commit version bump:"
echo "     git add pyproject.toml docs/source/conf.py"
echo "     git commit -m 'chore: bump version to $VERSION'"
echo ""
echo "  3. Create and push tag:"
echo "     git tag -a v$VERSION -m 'Release version $VERSION'"
echo "     git push origin main"
echo "     git push origin v$VERSION"
echo ""
echo "  4. Create GitHub release at:"
echo "     https://github.com/KIPAC/c2i2o/releases/new"
echo "     - Tag: v$VERSION"
echo "     - Title: v$VERSION"
echo "     - Copy content from CHANGELOG.md"
echo ""
echo "  5. GitHub Actions will automatically publish to PyPI"
echo ""
