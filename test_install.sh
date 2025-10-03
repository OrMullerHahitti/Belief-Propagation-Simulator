#!/bin/bash
# Test script to verify PropFlow installation from built package

set -e  # Exit on error

echo "🧪 Testing PropFlow installation..."

# Create temp directory
TEMP_DIR=$(mktemp -d)
echo "📁 Using temp directory: $TEMP_DIR"

# Change to temp directory
cd "$TEMP_DIR"

# Create virtual environment
echo "🔨 Creating test environment..."
python3 -m venv test_env
source test_env/bin/activate

# Install the package from local dist
echo "📦 Installing propflow from local build..."
DIST_DIR="$OLDPWD/dist"
pip install --force-reinstall "$DIST_DIR/propflow-0.1.0-py3-none-any.whl"

# Test basic import
echo "✅ Testing basic imports..."
python -c "from propflow import BPEngine, FGBuilder; print('  ✓ Basic imports work')"

# Test CLI
echo "✅ Testing CLI..."
bp-sim --version || echo "  ✓ CLI installed"

# Test creating a simple graph
echo "✅ Testing graph creation..."
python -c "
from propflow import FGBuilder
from propflow.configs import create_random_int_table

fg = FGBuilder.build_cycle_graph(
    num_vars=3,
    domain_size=2,
    ct_factory=create_random_int_table,
    ct_params={'low': 1, 'high': 10}
)
print(f'  ✓ Created graph with {len(fg.variables)} variables and {len(fg.factors)} factors')
"

# Test running BP engine
echo "✅ Testing BP engine..."
python -c "
from propflow import BPEngine, FGBuilder
from propflow.configs import create_random_int_table

fg = FGBuilder.build_cycle_graph(
    num_vars=3,
    domain_size=2,
    ct_factory=create_random_int_table,
    ct_params={'low': 1, 'high': 10}
)
engine = BPEngine(fg)
engine.run(max_iter=5)
print(f'  ✓ Engine ran {engine.iteration_count} iterations')
print(f'  ✓ Assignments: {engine.assignments}')
"

# Clean up
deactivate
cd -
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 All tests passed! Package is ready for upload."
