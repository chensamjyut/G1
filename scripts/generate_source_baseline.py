"""Explicit read-only original-vs-release baseline generator, no simulator.
The source checkout is needed ONLY for regenerating expected results.
"""
import argparse,json,hashlib,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--original-root',required=True,type=Path);p.add_argument('--release-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=a.release_root.resolve();t=a.original_root.resolve()
sys.path.insert(0,str(t/'src'))
import torch
from gimbal_tvr.monotone_stop_policy import MonotoneInference
torch.set_num_threads(4);torch.manual_seed(42)
m=MonotoneInference(str(r/'models/grpo_viewpoint.pt'),'cpu');b=json.loads((r/'tests/fixtures/baseline.json').read_text())
for row in b['cases']:
    row['expected']=m.predict(str(r/row['current']),str(r/row['target']),row['history'])
b['source_files']=[{'path':x['path'],'sha256':hashlib.sha256((t/x['path']).read_bytes()).hexdigest()} for x in b['source_files']]
b['torch']=torch.__version__;a.output.write_text(json.dumps(b,indent=2)+'\n')
