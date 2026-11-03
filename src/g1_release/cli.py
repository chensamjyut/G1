import argparse,json,sys,os,subprocess
from pathlib import Path
from .resources import verify_models,mission_config,verify_file

def deny_network():
    # Controlled prohibition used only for offline verification, not API execution.
    def audit(event,args):
        if event in ('socket.connect','socket.connect_ex','socket.getaddrinfo','socket.sendto'):
            raise RuntimeError('offline_network_forbidden:'+event)
    sys.addaudithook(audit)

def main():
    p=argparse.ArgumentParser(description='Single-point inspection release; simulator execution is explicit.')
    p.add_argument('--root',required=True,type=Path,help='Independent checkout/resource root, never inferred from cwd')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('check-models');sub.add_parser('environment')
    f=sub.add_parser('infer');f.add_argument('--current',required=True);f.add_argument('--target',required=True);f.add_argument('--history',default='[]');f.add_argument('--offline',action='store_true')
    f=sub.add_parser('replay');f.add_argument('--baseline',default='tests/fixtures/baseline.json');f.add_argument('--offline',action='store_true');f.add_argument('--output')
    f=sub.add_parser('preflight');f.add_argument('--config',required=True)
    f=sub.add_parser('run');f.add_argument('--config',required=True);f.add_argument('--sim-python',required=True)
    a=p.parse_args();root=a.root.resolve()
    if getattr(a,'offline',False):deny_network()
    if a.command=='environment':
        import importlib.metadata as md
        print(json.dumps({'python':sys.version,'executable':sys.executable,'isolated_venv':sys.prefix!=sys.base_prefix,'packages':{n:md.version(n) for n in ['torch','numpy','opencv-python-headless','Pillow','PyYAML']}},indent=2));return
    if a.command=='check-models':print(json.dumps(verify_models(root),indent=2));return
    if a.command in ('run','preflight'):
        cfg=mission_config(root,Path(a.config))
        if a.command=='preflight':print(json.dumps({'status':'FILES_VERIFIED','usd_dependency_check':'required before simulation','hardware_validated':False},indent=2));return
        script=root/'sim_runtime/run_single_point.py'
        env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        raise SystemExit(subprocess.call([a.sim_python,str(script),'--release-root',str(root),'--config',str(Path(a.config).resolve()),'--headless','--enable_cameras'],env=env,cwd=root))
    verify_models(root)
    import torch
    from g1_active_view.policies import OfflinePolicy
    torch.set_num_threads(4);torch.manual_seed(42)
    model=OfflinePolicy(root/'models/grpo_viewpoint.pt',root/'third_party/dinov2',root/'models/dinov2_vitb14_pretrain.pth','cpu')
    if a.command=='infer':
        out=model.predict(a.current,a.target,json.loads(a.history));out['task_success']=False;out['task_success_reason']='offline_policy_prediction_is_not_execution_or_reading';print(json.dumps(out,indent=2));return
    baseline=json.loads((root/a.baseline).read_text());results=[]
    for row in baseline['cases']:
        for role in ['current','target']:verify_file(root/row[role],row[role+'_sha256'])
        actual=model.predict(str(root/row['current']),str(root/row['target']),row['history']);expected=row['expected']
        logits=max(abs(x-y) for x,y in zip(actual['logits'],expected['logits']));probs=max(abs(x-y) for x,y in zip(actual['probabilities'],expected['probabilities']));sim=abs(actual['similarity']-expected['similarity'])
        passed=logits<=1e-4 and probs<=1e-5 and sim<=1e-5 and actual['action_id']==expected['action_id'] and actual['visual_inliers']==expected['visual_inliers']
        results.append({'case':row['name'],'passed':passed,'max_logits_abs_error':logits,'max_probability_abs_error':probs,'similarity_abs_error':sim,'action':actual['action_name'],'inliers':actual['visual_inliers']})
    out={'passed':all(r['passed'] for r in results),'cases':results,'network_prohibited':getattr(a,'offline',False),'device':'cpu','torch':torch.__version__,'tolerance_basis':'FP32 cross-PyTorch-version roundoff only; strict action/inlier equality; source same-environment parity also recorded'}
    if a.output:Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));raise SystemExit(0 if out['passed'] else 1)
if __name__=='__main__':main()
