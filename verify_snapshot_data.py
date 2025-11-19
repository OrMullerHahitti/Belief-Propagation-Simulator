import sys
from pathlib import Path
import warnings

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from propflow.snapshots.types import EngineSnapshot

def test_snapshot_data_deprecation():
    snapshot = EngineSnapshot(
        step=1,
        lambda_=0.5,
        dom={},
        N_var={},
        N_fac={},
        Q={},
        R={}
    )
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        data = snapshot.data
        print(f"Accessed snapshot.data: {data}")
        
        if len(w) > 0:
            print(f"Caught warning: {w[-1].message}")
            if issubclass(w[-1].category, DeprecationWarning):
                print("Verified: DeprecationWarning raised")
            else:
                print(f"Failed: Wrong warning type {w[-1].category}")
        else:
            print("Failed: No warning raised")

    if data is snapshot:
        print("Verified: snapshot.data returns self")
    else:
        print("Failed: snapshot.data does not return self")

if __name__ == "__main__":
    test_snapshot_data_deprecation()
