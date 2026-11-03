"""Explicit local, authorized staging; never downloads or publishes private assets.

Copies the USD dependency closure and rewrites only asset references in copies.
Source assets remain unchanged. Built-in MDL module names are preserved and
reported for validation inside Kit. Use standalone usd-core for this command.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
from pxr import Sdf,UsdUtils

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--source-root',required=True,type=Path);p.add_argument('--reader-python',required=True);a=p.parse_args();root=a.root.resolve();source=a.source_root.resolve()
    manifest=json.loads((root/'resources/external_manifest.json').read_text());dest=root/'resources/private';dest.mkdir(parents=True,exist_ok=True);bundle=dest/'usd';bundle.mkdir(exist_ok=True)
    mapping={};modules=set();roots={};records=[]
    for row in manifest:
        path=source/row['original_relative_path']
        if not path.is_file() or sha(path)!=row['sha256']:raise RuntimeError('source_resource_missing_or_changed:'+row['id'])
        roots[row['id']]=path.resolve()
    all_layers={};all_assets=set()
    for key in ('scene','robot'):
        layers,assets,missing=UsdUtils.ComputeAllDependencies(str(roots[key]))
        for name in missing:
            if name not in ('OmniPBR.mdl','OmniGlass.mdl','OmniSurface.mdl','gltf/pbr.mdl'):raise RuntimeError('unresolved_asset:'+name)
            modules.add(name)
        for layer in layers:all_layers[Path(layer.realPath).resolve()]=layer
        all_assets.update(Path(x).resolve() for x in assets)
    for path in sorted(set(all_layers)|all_assets):
        if not path.is_file():raise FileNotFoundError(str(path))
        # Deterministic name by content; preserve extension for texture plugins.
        mapping[path]=bundle/(sha(path)[:20]+'_'+path.name)
    for path,out in mapping.items():
        if path not in all_layers:shutil.copyfile(path,out)
        else:
            layer=all_layers[path];copy=Sdf.Layer.CreateAnonymous();copy.TransferContent(layer)
            def rewrite(value,layer=layer):
                if not value or value in modules:return value
                resolved=Path(Sdf.ComputeAssetPathRelativeToLayer(layer,value)).resolve()
                if resolved not in mapping:raise RuntimeError('dependency_not_in_closure:'+value)
                return './'+mapping[resolved].name
            UsdUtils.ModifyAssetPaths(copy,rewrite)
            if not copy.Export(str(out)):raise RuntimeError('USD_export_failed')
        records.append(dict(source_sha256=sha(path),path=str(out.relative_to(root)),sha256=sha(out),kind='USD reference relocation only' if path in all_layers else 'identical asset'))
    cfg=json.loads((root/'configs/mission.example.json').read_text())
    for key,field in [('scene','scene_path'),('robot','robot_usd')]:cfg[field]=str(mapping[roots[key]].relative_to(root))
    for key,field in [('walking_policy','walking_policy'),('target_image',None)]:
        path=roots[key];out=dest/(key+path.suffix);shutil.copyfile(path,out)
        records.append(dict(source_sha256=sha(path),path=str(out.relative_to(root)),sha256=sha(out),kind='identical resource'))
        if field:cfg[field]=str(out.relative_to(root))
        else:cfg['inspection_points'][0]['target_image_path']=str(out.relative_to(root))
    cfg['reader']['python']=str(Path(a.reader_python).resolve());cfg['resource_bundle']='resources/private/manifest.json'
    (dest/'manifest.json').write_text(json.dumps(dict(files=records,builtin_mdl_requires_kit_validation=sorted(modules),relocated=True,entrypoints={**{k:cfg[k] for k in ('scene_path','robot_usd','walking_policy')},'target_image_path':cfg['inspection_points'][0]['target_image_path']}),indent=2)+'\n')
    output=root/'configs/mission.local.json';output.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'config':str(output),'copied_files':len(records),'builtin_mdl_requires_kit_validation':sorted(modules),'published':False},indent=2))
if __name__=='__main__':main()
