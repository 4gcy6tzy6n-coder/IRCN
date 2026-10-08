from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from gated_runner import run
if __name__=='__main__':
    import json
    print(json.dumps(run(Path(__file__).resolve().parent.name),indent=2))
