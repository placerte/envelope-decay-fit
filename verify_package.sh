#!/bin/bash
# Quick verification script for envelope-decay-fit package

echo "=================================================="
echo "envelope-decay-fit v0.1.2 - Package Verification"
echo "=================================================="
echo ""

# Check Python version
echo "1. Python version:"
python --version
echo ""

# Check virtual environment
echo "2. Virtual environment:"
which python
echo ""

# Check package installation
echo "3. Package installation:"
python -c "from envelope_decay_fit import fit_piecewise_manual; print('✓ Package imported successfully')"
echo ""

# Check CLI command
echo "4. CLI command:"
which env-decay-fit
env-decay-fit --version
echo ""

# Run tests
echo "5. Running tests:"
pytest -q
echo ""

# Check example data
echo "6. Example data:"
ls data/envelope_exports/ | head -3
echo ""

# Test on one dataset
echo "7. Quick test on real data:"
env-decay-fit fit data/envelope_exports/free_plate_A3H1/hit_001.csv \
    --fn-hz 775.2 \
    --breakpoints "0.0,0.1,0.2" \
    --out-dir out/verification \
    2>&1 | grep -E "(Fit results saved|Storyboard plot)"
echo ""

echo "=================================================="
echo "✓ Package verification complete!"
echo "=================================================="
echo ""
echo "Package is ready for use."
echo "Documentation: README.md"
echo "Summary: PACKAGE_SUMMARY.md"
echo "Examples: examples/quickstart.py"
