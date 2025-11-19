import sys
import json
import os
from pathlib import Path

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from propflow.bp.engine_components import History

def test_history_json():
    # Create history without BCT (default)
    hist = History(engine_type="TestEngine", use_bct_history=False)
    
    # Mock some data
    # (In a real run, cycles would be populated)
    # We just want to see if to_json produces empty dict
    
    outfile = "test_history.json"
    hist.to_json(outfile)
    
    with open(outfile, "r") as f:
        content = json.load(f)
    
    print(f"JSON content: {content}")
    
    if content == {}:
        print("Issue reproduced: JSON is empty")
    else:
        print("JSON is not empty")

    if os.path.exists(outfile):
        os.remove(outfile)

if __name__ == "__main__":
    test_history_json()
