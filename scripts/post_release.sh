#!/bin/bash
# Script to run after a release

set -e

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: $0 <released-version>"
    echo "Example: $0 0.1.0"
    exit 1
fi

echo "=========================================="
echo "Post-Release Tasks for v$VERSION"
echo "=========================================="
echo ""

# Calculate next dev version
IFS='.' read -ra PARTS <<< "$VERSION"
MAJOR=${PARTS[0]}
MINOR=${PARTS[1]}
PATCH=${PARTS[2]}

NEXT_MINOR=$((MINOR + 1))
NEXT_VERSION="$MAJOR.$NEXT_MINOR.0"
DEV_VERSION="${NEXT_VERSION}-dev"

echo "Released version: $VERSION"
echo "Next dev version: $DEV_VERSION"
echo ""

# Update version to dev
echo "Updating version to $DEV_VERSION..."
sed -i.bak "s/^version = .*/version = \"$DEV_VERSION\"/" pyproject.toml
rm pyproject.toml.bak
echo "✓ Updated pyproject.toml to $DEV_VERSION"
echo ""

# Add unreleased section to CHANGELOG
echo "Updating CHANGELOG.md..."
DATE=$(date +%Y-%m-%d)

# Insert unreleased section after first heading
sed -i.bak "/^## \[Unreleased\]/a\\
\\
### Planned\\
- TBD\\
" CHANGELOG.md
rm CHANGELOG.md.bak

echo "✓ Updated CHANGELOG.md"
echo ""

echo "=========================================="
echo "Post-release complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Commit changes:"
echo "     git add pyproject.toml CHANGELOG.md"
echo "     git commit -m 'chore: bump version to $DEV_VERSION'"
echo "     git push origin main"
echo ""
echo "  2. Verify PyPI release:"
echo "     https://pypi.org/project/c2i2o/$VERSION/"
echo ""
echo "  3. Test installation:"
echo "     pip install c2i2o==$VERSION"
echo ""
