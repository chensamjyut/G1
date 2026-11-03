"""Submit START using the existing PID/nonce protocol, never fake ARRIVED/STOP."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ready',required=True);p.add_argument('--trigger',required=True);a=p.parse_args()
r=json.loads(Path(a.ready).read_text())
if not r.get('ready') or r['status']!='WAIT_FOR_START':raise SystemExit('mission_not_ready')
os.kill(r['pid'],0)
start=Path(f"/proc/{r['pid']}/stat").read_text().split()[21]
if start!=r['process_start_ticks']:raise SystemExit('stale_ready_pid')
payload={'command':'START','pid':r['pid'],'nonce':r['nonce']}
f=Path(a.trigger);tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(payload));tmp.replace(f)
print('START submitted; wait for actual navigation feedback')
