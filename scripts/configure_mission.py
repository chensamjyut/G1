"""Configure an independently obtained authorized resource bundle."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--reader-python',required=True);a=p.parse_args();r=a.root.resolve()
b=json.loads((r/'resources/private/manifest.json').read_text());c=json.loads((r/'configs/mission.example.json').read_text())
for k in ('scene_path','robot_usd','walking_policy'):c[k]=b['entrypoints'][k]
c['inspection_points'][0]['target_image_path']=b['entrypoints']['target_image_path'];c['resource_bundle']='resources/private/manifest.json';c['reader']['python']=str(Path(a.reader_python).resolve())
f=r/'configs/mission.local.json'
if f.exists():raise SystemExit('Existing mission.local.json preserved. Edit it or use a new checkout.')
f.write_text(json.dumps(c,indent=2)+'\n');print(f)
