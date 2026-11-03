"""Read-only USD composition check; use simulator Python, starts no SimulationApp."""
import argparse,json
from pathlib import Path
from pxr import UsdUtils
p=argparse.ArgumentParser();p.add_argument('usd');p.add_argument('--output');a=p.parse_args()
layers,assets,missing=UsdUtils.ComputeAllDependencies(str(Path(a.usd).resolve()))
r={'layer_count':len(layers),'asset_count':len(assets),'unresolved_count':len(missing),'passed':not missing}
if a.output:Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
raise SystemExit(0 if not missing else 2)
