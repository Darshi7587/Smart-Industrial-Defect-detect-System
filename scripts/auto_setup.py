# Auto-generate sample dataset without user input
import sys
sys.path.insert(0, '.')

# Patch input to auto-select option 4
import builtins
original_input = builtins.input
builtins.input = lambda x="": "4"

# Run the setup
exec(open('scripts/setup_dataset.py').read())
