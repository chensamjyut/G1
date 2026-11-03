from pathlib import Path
import hashlib,json,math

def sha256(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()

def tree_hash(root):
    root=Path(root); d=hashlib.sha256()
    for p in sorted((p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')),key=lambda p:p.relative_to(root).as_posix()):
        if p.is_symlink():raise ValueError('vendored_source_symlink:'+str(p))
        d.update(p.relative_to(root).as_posix().encode()+b'\0'+bytes.fromhex(sha256(p)))
    return d.hexdigest()

def verify_file(path,expected,size=None):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError('resource_missing:'+str(path))
    with path.open('rb') as f:
        if f.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):raise ValueError('LFS_pointer_not_downloaded:'+str(path))
    actual=sha256(path)
    if actual!=expected:raise ValueError('sha256_mismatch:'+str(path))
    if size is not None and path.stat().st_size!=size:raise ValueError('size_mismatch:'+str(path))
    return {'path':str(path),'sha256':actual,'size_bytes':path.stat().st_size}

def verify_models(root):
    root=Path(root).resolve();m=json.loads((root/'models/manifest.json').read_text());out=[]
    for r in m['models']:out.append(verify_file(root/r['path'],r['sha256'],r['size_bytes']))
    d=m['dinov2_source'];actual=tree_hash(root/d['path'])
    if actual!=d['tree_sha256']:raise ValueError('dinov2_source_tree_mismatch')
    return {'models':out,'dinov2_tree_sha256':actual}

def mission_config(root,path):
    root=Path(root).resolve();cfg=json.loads(Path(path).read_text())
    def resolve(value,field):
        if not isinstance(value,str) or not value.strip():raise ValueError('unconfigured:'+field)
        p=Path(value).expanduser();return str((root/p).resolve())
    for key in ('scene_path','robot_usd','walking_policy','execution_root','output_root'):cfg[key]=resolve(cfg.get(key),key)
    cfg['reader']['python']=resolve(cfg['reader'].get('python'),'reader.python')
    if not cfg['reader'].get('model'):raise ValueError('unconfigured:reader.model')
    if len(cfg['inspection_points'])!=1:raise ValueError('single_inspection_point_required')
    target=cfg['inspection_points'][0];target['target_image_path']=resolve(target.get('target_image_path'),'target_image')
    for key in ('checkpoint','dino_source','dino_weights'):cfg['viewpoint_policy'][key]=resolve(cfg['viewpoint_policy'].get(key),'viewpoint_policy.'+key)
    verify_models(root)
    verify_file(cfg['viewpoint_policy']['checkpoint'],cfg['viewpoint_policy']['checkpoint_sha256'])
    verify_file(target['target_image_path'],cfg['target_metadata']['sha256'])
    if cfg.get('resource_bundle'):
        bundle=json.loads((root/cfg['resource_bundle']).read_text())
        verified=set()
        for row in bundle['files']:
            verify_file(root/row['path'],row['sha256']);verified.add(str((root/row['path']).resolve()))
        for key in ('scene_path','robot_usd','walking_policy'):
            if cfg[key] not in verified:raise ValueError('resource_not_in_verified_bundle:'+key)
    else:
        manifest=json.loads((root/'resources/external_manifest.json').read_text())
        keys={'scene':'scene_path','robot':'robot_usd','walking_policy':'walking_policy'}
        for r in manifest:
            if r['id'] in keys:verify_file(cfg[keys[r['id']]],r['sha256'])
    if not Path(cfg['reader']['python']).is_file():raise FileNotFoundError('reader_python_missing')
    # Unknown USD composition dependencies are validated by the explicit USD checker before launch.
    return cfg
