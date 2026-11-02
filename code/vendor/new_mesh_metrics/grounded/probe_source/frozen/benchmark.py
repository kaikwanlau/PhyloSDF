"""Resumable, bounded matched decoder smoke benchmark. No research writes."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def atomic_json(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,allow_nan=False)); t.replace(p)

def extract(field, extractor='auto'):
    import numpy as np
    if not np.isfinite(field).all(): raise ValueError('nonfinite SDF')
    if not float(field.min()) < 0 < float(field.max()): raise ValueError('no strict zero crossing')
    r=field.shape[0]; spacing=2/(r-1)
    if extractor=='auto': extractor='skimage' if importlib.util.find_spec('skimage') else 'vtk'
    if extractor=='skimage':
        from skimage.measure import marching_cubes
        v,f,_,_=marching_cubes(field,level=0,spacing=(spacing,)*3,allow_degenerate=False)
        v=v-1
    else:
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkFiltersCore import vtkMarchingCubes
        from vtkmodules.util.numpy_support import numpy_to_vtk,vtk_to_numpy
        image=vtkImageData(); image.SetDimensions(r,r,r); image.SetOrigin(-1,-1,-1); image.SetSpacing(spacing,spacing,spacing)
        image.GetPointData().SetScalars(numpy_to_vtk(field.ravel(order='F'),deep=True))
        mc=vtkMarchingCubes(); mc.SetInputData(image); mc.SetValue(0,0); mc.Update()
        poly=mc.GetOutput()
        if not poly.GetNumberOfPoints(): raise ValueError('zero contour empty')
        v=vtk_to_numpy(poly.GetPoints().GetData()).copy()
        f=vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1,4)[:,1:].copy()
    if len(f)==0: raise ValueError('empty faces')
    return np.asarray(v),np.asarray(f,dtype=np.int32),extractor

def mesh_info(v,f):
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]); e=np.sort(e,axis=1)
    unique,counts=np.unique(e,axis=0,return_counts=True)
    graph=coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(len(v),len(v)))
    components=connected_components(graph,directed=False,return_labels=False)
    tri=v[f]; areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    return {'vertices':len(v),'faces':len(f),'connected_components':int(components),
            'boundary_contact_vertices':int(np.any(np.isclose(np.abs(v),1,atol=2e-6),axis=1).sum()),
            'open_edges':int((counts==1).sum()),'nonmanifold_edges':int((counts>2).sum()),
            'degenerate_faces':int((areas<=1e-14).sum()),'area':float(areas.sum()),
            'bounds':[v.min(0).tolist(),v.max(0).tolist()], 'component_filtering':'none'}

def sample_surface(v,f,n,seed):
    import numpy as np
    rng=np.random.default_rng(seed); tri=v[f].astype(np.float64)
    areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    t=tri[rng.choice(len(f),n,p=areas/areas.sum())]
    u=np.sqrt(rng.random(n)); w=rng.random(n)
    return (1-u[:,None])*t[:,0]+(u*(1-w))[:,None]*t[:,1]+(u*w)[:,None]*t[:,2]

def cd(a,b):
    from scipy.spatial import cKDTree
    return float((cKDTree(a).query(b,workers=1)[0]**2).mean()+(cKDTree(b).query(a,workers=1)[0]**2).mean())

def self_check(extractor):
    import numpy as np
    r=25; x=np.linspace(-1,1,r,dtype=np.float32)
    # Unequal axis scales and off-centre position catch transposition/origin defects.
    field=((x[:,None,None]-.2)/.5)**2+((x[None,:,None]+.1)/.3)**2+(x[None,None,:]/.7)**2-1
    v,f,e=extract(field,extractor)
    expected=np.array([[-.3,-.4,-.7],[.7,.2,.7]])
    error=float(np.abs(np.array([v.min(0),v.max(0)])-expected).max())
    assert error<2/(r-1),error
    try: extract(np.ones((5,5,5),np.float32),extractor)
    except ValueError: pass
    else: raise AssertionError('no-crossing volume accepted')
    return {'extractor':e,'analytic_ellipsoid_bound_error':error,'no_crossing_rejected':True}

def make_decoder(ckpt,backend,source,device):
    import numpy as np
    from runtime_backend import NumpyDecoder
    if backend=='numpy': return NumpyDecoder(ckpt), 'cpu'
    import torch
    spec=importlib.util.spec_from_file_location('original_deepsdf',source)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg=ckpt['config']; keys=['latent_dim','hidden_dim','num_layers','skip_connection','weight_norm']
    model=mod.DeepSDFDecoder(**{k:cfg[k] for k in keys if k in cfg})
    model.load_state_dict({k:torch.from_numpy(np.array(v,copy=True)) for k,v in ckpt['decoder_state_dict'].items()})
    model.to(device);model.eval()
    class Predictor:
        def predict(self,points,code):
            with torch.inference_mode():
                p=torch.as_tensor(points,dtype=torch.float32,device=device)
                z=torch.as_tensor(code,dtype=torch.float32,device=device).expand(len(p),-1)
                return model(z,p).reshape(-1).cpu().numpy()
    predictor=Predictor()
    # Check NumPy implementation on the actual loaded torch weights before reuse.
    pts=np.random.default_rng(20261014).uniform(-1,1,(127,3)).astype(np.float32)
    z=ckpt['latent_lib_state_dict']['latent_codes.weight'][0]
    diff=float(np.max(np.abs(predictor.predict(pts,z)-NumpyDecoder(ckpt).predict(pts,z))))
    if diff>3e-5: raise RuntimeError(f'NumPy/torch parity failed: {diff}')
    predictor.parity_max_abs=diff
    return predictor,device

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resolution',type=int,default=32);p.add_argument('--chunk',type=int,default=2048);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--backend',choices=['numpy','torch','auto'],default='auto');p.add_argument('--extractor',choices=['auto','skimage','vtk'],default='auto')
    p.add_argument('--reconstruction-points',type=int,default=8192);a=p.parse_args()
    if a.resolution<8 or a.chunk<1 or not 1<=a.threads<=8: p.error('invalid bounded settings')
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VTK_SMP_MAX_THREADS']:os.environ[k]=str(a.threads)
    import numpy as np
    from runtime_backend import load_checkpoint
    a.root=a.root.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    research=a.root/'revision_reading';assert not a.out.is_relative_to(research)
    backup=research/'drive_backup';start=time.time();backend=a.backend
    if backend=='auto': backend='torch' if importlib.util.find_spec('torch') else 'numpy'
    device='cpu'
    if backend=='torch':
        import torch
        torch.set_num_threads(a.threads)
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ckpts={k:load_checkpoint(backup/f'checkpoint_metadata/deepsdf/{k}.pt') for k in ['best','final']}
    models={k:make_decoder(c,backend,backup/'models/deepsdf.py',device)[0] for k,c in ckpts.items()}
    check=self_check(a.extractor)
    metadata=json.loads((backup/'processed/metadata.json').read_text());specs=metadata['specimens']
    ids=[next(i for i,s in enumerate(specs) if s['filename']==name) for name in ['G.FortisA','G.FortisB']]
    codes={k:np.asarray(c['latent_lib_state_dict']['latent_codes.weight']) for k,c in ckpts.items()}
    ctrlpath=a.root/'working_results/startup/controls/control_codes.npz'
    if not ctrlpath.exists():ctrlpath=a.root/'outputs/revision/cpu_controls/results/control_codes.npz'
    controls=np.load(ctrlpath,allow_pickle=False)
    sid=specs[ids[0]]['species_idx']
    tasks=[]
    for dec in ['best','final']:
        for i in ids:tasks.append((f'{dec}_paired_{specs[i]["filename"]}',dec,codes[dec][i], 'paired_real',i))
        for name in ['empirical','gaussian']:
            key=next(k for k in controls.files if name in k and controls[k].ndim==2)
            labelkey=next(k for k in controls.files if ('species' in k or 'label' in k) and controls[k].ndim==1)
            j=int(np.flatnonzero(controls[labelkey]==sid)[0])
            tasks.append((f'{dec}_{name}_species{sid}_0',dec,controls[key][j].astype(np.float32),name,None))
    for i in ids:tasks.append((f'best_historical_finalcode_{specs[i]["filename"]}','best',codes['final'][i],'historical_pairing',i))
    inputs=[backup/'checkpoint_metadata/deepsdf/best.pt',backup/'checkpoint_metadata/deepsdf/final.pt',backup/'processed/metadata.json',backup/'arrays/processed/G.FortisA.npz',backup/'models/deepsdf.py',ctrlpath,Path(__file__),Path(__file__).with_name('runtime_backend.py')]
    versions={k:importlib.metadata.version(k) for k in ['numpy','scipy',check['extractor'] if check['extractor']=='vtk' else 'scikit-image']}
    if backend=='torch':versions['torch']=importlib.metadata.version('torch')
    config={'resolution':a.resolution,'chunk':a.chunk,'threads':a.threads,'backend':backend,'device':device,'extractor':check['extractor'],
            'seed':20261014,'reconstruction_points':a.reconstruction_points,'versions':versions,'inputs':{str(q.relative_to(a.root)):sha(q) for q in inputs},
            'tasks':[{ 'id':t[0],'decoder':t[1],'code_sha256':hashlib.sha256(np.ascontiguousarray(t[2]).tobytes()).hexdigest(),'group':t[3],'specimen_index':t[4]} for t in tasks]}
    lock=a.out/'run_config.json'
    if lock.exists() and json.loads(lock.read_text())!=config:raise RuntimeError('Resume configuration/input hash mismatch. Use a new run folder.')
    atomic_json(lock,config);atomic_json(a.out/'extractor_self_check.json',check)
    import shutil
    for q in inputs:
        if q.suffix=='.py':
            dest=a.out/'source_snapshot'/q.relative_to(a.root);dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(q,dest)
            if sha(dest)!=sha(q):raise RuntimeError('Source snapshot mismatch')
    r=a.resolution;axis=np.linspace(-1,1,r,dtype=np.float32)
    records=[]
    for name,dec,z,group,i in tasks:
        task=a.out/name;task.mkdir(exist_ok=True);done=task/'result.json'
        if done.exists():
            rec=json.loads(done.read_text())
            for file,h in rec.get('output_sha256',{}).items():
                if sha(task/file)!=h:raise RuntimeError('Saved task hash mismatch: '+name)
            records.append(rec);print('RESUMED',name,flush=True);continue
        progress=task/'progress.json';volume=task/'sdf.npy'
        saved=json.loads(progress.read_text()) if progress.exists() else {'next_index':0,'chunks':[]}
        offset=saved['next_index']
        if type(offset) is not int or not 0<=offset<=r**3:raise RuntimeError('Invalid resume offset')
        if offset and not saved.get('chunks'):raise RuntimeError('Partial resume lacks verified chunk ledger')
        field=np.lib.format.open_memmap(volume,mode='r+' if offset else 'w+',dtype=np.float32,shape=(r,r,r))
        if field.shape!=(r,r,r) or field.dtype!=np.float32:raise RuntimeError('SDF resume shape/dtype mismatch')
        cursor=0
        for piece in saved.get('chunks',[]):
            if piece['start']!=cursor or not cursor<piece['end']<=offset:raise RuntimeError('Partial ledger not contiguous')
            block=np.ascontiguousarray(field.reshape(-1)[cursor:piece['end']])
            if hashlib.sha256(block.tobytes()).hexdigest()!=piece['sha256']:raise RuntimeError('Partial SDF checksum mismatch')
            cursor=piece['end']
        if cursor!=offset:raise RuntimeError('Partial ledger does not reach saved offset')
        timer=time.time();previous=json.loads(progress.read_text()).get('compute_seconds',0) if progress.exists() else 0
        for lo in range(offset,r**3,a.chunk):
            hi=min(lo+a.chunk,r**3);idx=np.arange(lo,hi)
            pts=np.column_stack([axis[idx//(r*r)],axis[(idx//r)%r],axis[idx%r]])
            pred=models[dec].predict(pts,z);field.reshape(-1)[lo:hi]=pred;field.flush()
            saved['chunks'].append({'start':lo,'end':hi,'sha256':hashlib.sha256(np.ascontiguousarray(pred,dtype=np.float32).tobytes()).hexdigest()})
            atomic_json(progress,{'next_index':hi,'compute_seconds':previous+time.time()-timer,'chunks':saved['chunks']})
        query_seconds=previous+time.time()-timer;exstart=time.time()
        rec={'task':name,'decoder':dec,'group':group,'specimen_index':i,'status':'completed','resolution':r,'zero_level':0,
             'spacing':2/(r-1),'query_seconds':query_seconds,'sdf_min':float(field.min()),'sdf_max':float(field.max()),
             'process_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'grid_frame':'decoder normalized [-1,1]^3','scope':'reduced-grid CPU/GPU smoke test, not publication comparison'}
        try:
            v,f,ex=extract(field,a.extractor); rec.update(mesh_info(v,f));rec['extractor']=ex
            np.savez_compressed(task/'mesh.npz',vertices=v,faces=f)
            rec['output_sha256']={'mesh.npz':sha(task/'mesh.npz'),'sdf.npy':sha(volume)}
        except ValueError as e:
            rec['status']='mesh_failed';rec['failure']=str(e);rec['output_sha256']={'sdf.npy':sha(volume)}
        rec['extraction_seconds']=time.time()-exstart
        rec['process_peak_rss_MiB_after_extraction']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        atomic_json(done,rec);records.append(rec)
        with (a.out/'execution_log.jsonl').open('a') as log:log.write(json.dumps({'event':'decode_task',**rec})+'\n')
        print(name,rec['status'],round(query_seconds,3),'sec',flush=True)
    # Fixed SDF reconstruction cohort; this is IN-TRAINING data from one specimen.
    reconstruction=[];q=backup/'arrays/processed/G.FortisA.npz';data=np.load(q,allow_pickle=False)
    rng=np.random.default_rng(20261014);ix=np.sort(rng.choice(len(data['points']),min(a.reconstruction_points,len(data['points'])),replace=False))
    pts=data['points'][ix].astype(np.float32);gt=data['sdf'][ix].reshape(-1)
    for dec,codearm in [('best','best'),('final','final'),('best','final'),('final','best')]:
        pred=np.concatenate([models[dec].predict(pts[j:j+a.chunk],codes[codearm][ids[0]]) for j in range(0,len(pts),a.chunk)])
        reconstruction.append({'decoder':dec,'code_checkpoint':codearm,'n':len(ix),'clamped_MAE':float(np.abs(np.clip(pred,-.1,.1)-np.clip(gt,-.1,.1)).mean()),'unclamped_MAE':float(np.abs(pred-gt).mean()),'sign_accuracy':float(((pred<0)==(gt<0)).mean())})
    np.save(a.out/'reconstruction_indices.npy',ix)
    atomic_json(a.out/'reconstruction.json',{'specimen':'G.FortisA','npz_sha256':sha(q),'seed':20261014,'results':reconstruction,
       'interpretation':'one training specimen, not independent generalization; provisional pairing evidence only'})
    # Deterministic area sampling in unchanged decoder frame, method-separated.
    clouds={}
    for rec in records:
        if rec['status']=='completed':
            q=a.out/rec['task']/'mesh.npz';mesh=np.load(q,allow_pickle=False)
            clouds[rec['task']]=sample_surface(mesh['vertices'],mesh['faces'],10000,int(sha(q)[:8],16))
    distances=[]
    for dec in ['best','final']:
        names=[f'{dec}_paired_G.FortisA',f'{dec}_paired_G.FortisB']
        if all(n in clouds for n in names):distances.append({'kind':'paired_real_code_A_vs_B','decoder':dec,'CD_squared_symmetric_sum':cd(*[clouds[n] for n in names]),'n_points_each':10000})
    for group in ['empirical','gaussian']:
        names=[f'{dec}_{group}_species{sid}_0' for dec in ['best','final']]
        if all(n in clouds for n in names):distances.append({'kind':'same_control_code_decoder_sensitivity','control':group,'CD_squared_symmetric_sum':cd(*[clouds[n] for n in names]),'n_points_each':10000})
    atomic_json(a.out/'mesh_diagnostics.json',{'results':distances,'units':'normalized squared-coordinate','normalization':'none; retain decoder frame','limit':'surface sampling noise included; 1 control per arm; no diversity ranking'})
    summary={'status':'completed','tasks':len(records),'valid_meshes':sum(x['status']=='completed' for x in records),'mesh_failures':sum(x['status']!='completed' for x in records),
             'elapsed_seconds_this_invocation':time.time()-start,'records':records,'backend':backend,'device':device,
             'process_cumulative_peak_rss_MiB_full_pipeline':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             'torch_parity_max_abs':{k:getattr(v,'parity_max_abs',None) for k,v in models.items()},
             'reference_selection':'provisional only; one training NPZ and reduced grid insufficient to lock best/final globally',
             'not_completed':['full-resolution evaluation','flow training','matched augmented flow/control experiment','independent biological validation']}
    atomic_json(a.out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
    return 0

if __name__=='__main__':sys.exit(main())
