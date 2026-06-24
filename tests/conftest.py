import sys
from pathlib import Path

# Allow `SeedDataGen.*` imports regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
