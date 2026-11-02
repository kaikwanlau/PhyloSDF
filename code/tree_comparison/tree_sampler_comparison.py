"""Matched sampler comparison conditional on the existing all100 dated-tree DeepSDF.

New prospective protocol, 24 October 2026; no Stage-1 training and no old output edits.
Use preflight(svc, base=...) before run(svc, folder_id=..., base=...).
The companion rep_helper.py preserves the recovered scientific calculations;
private storage locators are redacted. Historical entry points are not called here.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib, importlib.util, json, platform, random, shutil, sys, time, zipfile
from pathlib import Path
import numpy as np

BASE = Path('/content/PHYLOSDF_TREE_COMPARISON_20261024')
OUTPUT_FOLDER = 'local-input-3bfbceb15820db22'
SEEDS = (42, 43, 44)
METHODS = ('tree_pooled_residual_flow', 'tree_pooled_gaussian_residual',
           'tree_empirical_residual', 'tree_per_species_gaussian')
ARCHIVES = {
    'loss_ablation.zip': ('local-input-459477aa0330855c', '3b4e9d03aebaeaa2ff3039006f4f46719435679048c1545d2c34f30b11903e0f', 'tree_results'),
    'matched_inputs.zip': ('local-input-4547bb9fdfe0aaff', '93f2dc31651f900981076010a6a039bd84314434c41270fbdf64e97551cef169', 'matched'),
    'mesh_inputs.zip': ('local-input-84ff5aba929ef1c4', '30e17ce1efa4a478e5253b29f9885f4b2f3999a1b1ea27599550dc2220577eb7', 'mesh'),
}
TREE_DECODER_SHA = '4ed5a0bcd13bc1ee38bc3da0b8bbfce133e6096635a2e8837c27c8c10317c44f'
PROTOCOL = {
    'schema': 'phylosdf_tree_primary_samplers_20261024_v1',
    'representation': 'existing beta_tree final checkpoint; all100 skulls, 24 species, seed42, 2000 epochs',
    'representation_retrained': False,
    'decoder_sha256': TREE_DECODER_SHA,
    'seeds': list(SEEDS), 'methods': list(METHODS), 'latent_dim': 64,
    'training': {'updates': 10000, 'batch_size': 100, 'replacement': True,
                 'conditioning': 'constant null species token, zero size, zero phylogeny',
                 'optimizer': 'AdamW', 'learning_rate': 1e-4, 'weight_decay': 1e-5,
                 'cosine_eta_min': 1e-6, 'gradient_clip_norm': 1., 'variance_weight': .1,
                 'dropout': .05, 'no_augmentation': True,
                 'variance': 'scalar unbiased variance over all minibatch entries; midpoint one-step surrogate'},
    'generation': {'temperature': 1., 'Euler_steps': 50, 'guidance': 1.,
                   'slots': '140: 10 each of four focal species followed by100 mixture slots, index modulo24',
                   'Gaussian_noise': 'common torch CPU standard-normal float32 noise, seed+100000',
                   'empirical_donors': 'uniform real100 residuals, NumPy default_rng(seed+500000)'},
    'pooled_gaussian': 'new explicitly specified full-covariance fit: uniform empirical population residual covariance, no jitter; symmetric PSD square root. Historical main100 generator bytes were not recovered.',
    'per_species_gaussian': 'original vendor PhyloGMM covariance: global full-code unbiased covariance+1e-4I; alpha2/(n+2) for n>=3; otherwise half global. Shared-noise symmetric PSD root preserves distribution, not historical random draws.',
    'evaluation': {'resolution': 256, 'domain': [-1., 1.], 'spacing': 2/255,
                   'decoder': 'frozen FP32, TF32 off', 'components': 'all, no repair',
                   'points_per_mesh': 10000, 'surface_seeds': [20261014, 20261015],
                   'metric_source': 'unchanged vendor_metrics.py from verified mesh archive; exact KD-tree nearest distances',
                   'reference': 'same100 raw skulls; species-balanced MMD/MorphFID; all attempts retained'},
    'scope': 'sampler comparison conditional on one all100 tree representation; generation seeds do not replicate representation training; training-reference evaluation, not unseen-specimen generalization',
    'processed_inputs': 'original100 processed reference NPZ files downloaded by their locked Drive IDs and verified SHA256; frozen evaluator reads saved centroid, scale, vertex and face counts. No dense SDF fields downloaded.',
}
_rep = None
_svc = None

def log(message):
    print(time.strftime('[%Y-%m-%d %H:%M:%S UTC] ', time.gmtime())+str(message), flush=True)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def configure(base=BASE,svc=None):
    global BASE,_rep,_svc
    BASE=Path(base);BASE.mkdir(parents=True,exist_ok=True)
    if svc is not None:_svc=svc
    if _rep is None:_rep=load_module(Path(__file__).with_name('rep_helper.py'),'tree_comparison_rep_helper')
    _rep.ROOT=BASE;_rep.DL=BASE/'downloads';_rep.SRC=BASE/'matched/matched_cfm'
    _rep.MESHSRC=BASE/'mesh/new_mesh_metrics';_rep.REFS=BASE/'refs'
    _rep.WORK=BASE/'work';_rep.OUT=BASE/'results';_rep.STUDY_GROUP=PROTOCOL['schema']
    _rep.BUDGET['rows_per_method']=None
    return _rep

def download(file_id,path):
    if _svc is None:raise RuntimeError('Pass authenticated Drive svc from Colab.')
    from googleapiclient.http import MediaIoBaseDownload
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.part')
    with tmp.open('wb') as f:
        d=MediaIoBaseDownload(f,_svc.files().get_media(fileId=file_id,supportsAllDrives=True),chunksize=16<<20)
        done=False
        while not done:_,done=d.next_chunk(num_retries=5)
    tmp.replace(path)

def safe_extract(zpath,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():continue
            target=(destination/info.filename).resolve()
            if destination not in target.parents:raise RuntimeError('Unsafe archive member')
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst)

def lock_protocol(rep):
    src={str(p.relative_to(BASE)):sha(p) for p in [rep.SRC/'run_matched_cfm.py',rep.SRC/'vendor/diffusion.py',
        rep.MESHSRC/'vendor/vendor_metrics.py',BASE/'tree_results/work/beta_tree/deepsdf/final.pt',
        BASE/'tree_results/work/beta_tree/deepsdf/CODES.npz']}
    src.update({'tree_sampler_comparison.py':sha(__file__),'rep_helper.py':sha(Path(__file__).with_name('rep_helper.py'))})
    lock={'protocol':PROTOCOL,'sources':src}
    key=hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest()
    lock['protocol_and_source_sha256']=key
    p=BASE/'PROTOCOL_LOCK.json'
    if p.exists() and json.loads(p.read_text())!=lock:raise RuntimeError('Protocol/source changed in existing experiment root. Use a new root.')
    write_json(p,lock);return key

def restore_archives(rep):
    for name,(fid,digest,destination) in ARCHIVES.items():
        path=BASE/'downloads'/name
        if not path.exists():download(fid,path)
        if sha(path)!=digest:raise RuntimeError('Archive SHA mismatch: '+name)
        marker=BASE/destination/'.verified_archive_sha256'
        if not marker.exists():
            safe_extract(path,BASE/destination);marker.write_text(digest)
        elif marker.read_text()!=digest:raise RuntimeError('Restored source identity changed')

def inputs(rep):
    ckdir=BASE/'tree_results/work/beta_tree/deepsdf'
    complete=json.loads((ckdir/'TRAINING_COMPLETE.json').read_text())
    train=json.loads((ckdir/'TRAIN_LOG.json').read_text())
    if sha(ckdir/'final.pt')!=TREE_DECODER_SHA or complete['final_sha256']!=TREE_DECODER_SHA:raise RuntimeError('Wrong paired tree decoder')
    if complete['epochs']!=2000 or train['spec']!={'beta':.005,'matrix':'tree','specimens':'all'}:raise RuntimeError('Wrong Stage1 training provenance')
    with np.load(ckdir/'CODES.npz',allow_pickle=False) as f:
        codes=f['codes'].astype(np.float32);species=f['species'].astype(np.int64)
        indices=f['specimen_index'];names=f['specimen'].astype(str).tolist()
    if codes.shape!=(100,64) or not np.isfinite(codes).all() or not np.array_equal(indices,np.arange(100)):raise RuntimeError('Invalid all100 code library')
    meta_names,meta_sp=rep.specimens()
    if names!=meta_names or not np.array_equal(species,meta_sp):raise RuntimeError('Specimen/species order mismatch')
    matrix=rep.matrix('tree')
    if hashlib.sha256(json.dumps(rep.TREE_MYR,separators=(',',':')).encode()).hexdigest()!=rep.TREE_SHA:raise RuntimeError('Tree matrix changed')
    backend=load_module(rep.MESHSRC/'grounded/probe_source/frozen/runtime_backend.py','tree_preflight_backend')
    ck=backend.load_checkpoint(ckdir/'final.pt')
    native=np.asarray(ck['latent_lib_state_dict']['latent_codes.weight'],np.float32)
    if not np.array_equal(codes,native):raise RuntimeError('Decoder-native latent codes do not match CODES.npz')
    lpc=rep.pair_correlation(codes,species,matrix)
    if lpc<.98:raise RuntimeError('Unexpected tree-distance agreement')
    return ckdir/'final.pt',codes,species,{'code_shape':codes.shape,'tree_LPC':lpc,'tree_sha256':rep.TREE_SHA,
        'paired_decoder_sha256':TREE_DECODER_SHA,'native_code_parity_max_abs':float(np.max(np.abs(native-codes))),
        'Stage1_training_updates':train['updates'],'Stage1_training_epochs':complete['epochs']}

def prepare_references(rep):
    """Restore original raw meshes and original saved normalization records without altering manifests."""
    source=rep.MESHSRC/'vendor/reference_manifest.json'
    original=json.loads(source.read_text()); checks=[]
    for i,row in enumerate(original['references']):
        for pathkey,idkey,shakey in [('raw_stl','raw_drive_id','raw_sha256'),
                                      ('processed_npz','processed_drive_id','processed_sha256')]:
            p=rep.REFS/row[pathkey]
            if not p.exists() or sha(p)!=row[shakey]:download(row[idkey],p)
            if sha(p)!=row[shakey]:raise RuntimeError('Reference SHA mismatch: '+row[pathkey])
        checks.append({'specimen':row['specimen'],'raw_sha256':row['raw_sha256'],
                       'processed_sha256':row['processed_sha256'],'original_bytes_verified':True})
        if (i+1)%10==0:log(f'Original reference files verified {i+1}/100 skulls')
    rep.REFS.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,rep.REFS/'reference_manifest.json')
    shutil.copyfile(source,rep.REFS/'original_reference_manifest.json')
    write_json(BASE/'results/REFERENCE_TRANSFORM_CHECKS.json',checks)


def psd_root(covariance):
    c=(np.asarray(covariance,np.float64)+np.asarray(covariance,np.float64).T)/2
    w,v=np.linalg.eigh(c);tol=1e-10*max(1.,float(np.linalg.norm(c,ord=2)))
    if w.min()<-tol:raise RuntimeError('Invalid covariance')
    root=(v*np.sqrt(np.maximum(w,0)))@v.T
    if not np.allclose(root@root.T,c,rtol=1e-9,atol=tol):raise RuntimeError('PSD root covariance mismatch')
    return root

def simple_samples(rep,codes,species,seed):
    stats=rep.flow_stats(codes,species);centroids=np.asarray(stats['centroids'],np.float64)
    residual=codes.astype(np.float64)-centroids[species];mean=residual.mean(0)
    centered=residual-mean;cov=centered.T@centered/len(residual)
    requested,cohort,slot,noise=rep.generation_inputs(seed)
    donors=np.random.default_rng(seed+500000).integers(0,len(codes),size=len(requested))
    pooled=(centroids[requested]+mean+noise.astype(np.float64)@psd_root(cov).T).astype(np.float32)
    empirical=(centroids[requested]+residual[donors]).astype(np.float32)
    gmm=rep.rmc().load_vendor('diffusion').PhyloGMM();gmm.fit(codes,species)
    per=np.stack([gmm.means[int(s)].astype(np.float64)+noise[i].astype(np.float64)@psd_root(gmm.covs[int(s)]).T for i,s in enumerate(requested)]).astype(np.float32)
    arrays={METHODS[1]:pooled,METHODS[2]:empirical,METHODS[3]:per}
    if any(a.shape!=(140,64) or not np.isfinite(a).all() for a in arrays.values()):raise RuntimeError('Bad baseline samples')
    if not np.allclose(empirical.astype(np.float64)-centroids[requested],residual[donors],atol=1e-7):raise RuntimeError('Empirical donor reconstruction failed')
    return arrays,{'species':requested,'cohort':cohort,'sample_slot':slot,'noise':noise,'empirical_donor_indices':donors,
        'pooled_residual_mean':mean,'pooled_residual_population_covariance':cov,
        'pooled_covariance_unbiased_sensitivity_multiplier':len(codes)/(len(codes)-1)}

def train_flow(rep,codes,species,seed,updates=10000):
    import torch
    m=rep.rmc();m.deterministic();stats=rep.flow_stats(codes,species)
    targets=(codes-stats['centroids'][species]-stats['residual_mean'])/stats['residual_global_std']
    n=len(codes);batch_size=100;out=BASE/f'work/seed{seed}/flow';out.mkdir(parents=True,exist_ok=True)
    m.seed_all(seed);initial=m.load_vendor('diffusion').PhyloFlowNet(**m.model_kwargs(32)).state_dict()
    m.seed_all(seed);model=m.load_vendor('diffusion').PhyloFlowNet(**m.model_kwargs(32)).to('cuda');model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=updates,eta_min=1e-6)
    target=torch.as_tensor(targets);weights=torch.full((n,),3.,dtype=torch.float64)
    sampler=torch.Generator(device='cpu').manual_seed(seed+200000)
    flow=torch.Generator(device='cuda').manual_seed(seed+300000);m.seed_all(seed+400000)
    trace=[];start=1;resume=out/'resume.pt';final=out/'final.pt';key=json.loads((BASE/'PROTOCOL_LOCK.json').read_text())['protocol_and_source_sha256']
    if final.exists():
        if not final.with_suffix('.sha256').exists() or final.with_suffix('.sha256').read_text().strip()!=sha(final):raise RuntimeError('Final flow checksum mismatch')
        saved=torch.load(final,map_location='cuda',weights_only=False)
        if saved['protocol_sha256']!=key or saved['update']!=updates:raise RuntimeError('Final flow identity mismatch')
        model.load_state_dict(saved['model']);model.eval();return model,stats,saved['trace']
    if resume.exists():
        digest=resume.with_suffix('.sha256')
        if not digest.exists() or digest.read_text().strip()!=sha(resume):raise RuntimeError('Flow resume checksum mismatch')
        saved=torch.load(resume,map_location='cuda',weights_only=False)
        if saved['protocol_sha256']!=key:raise RuntimeError('Resume protocol mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        sampler.set_state(saved['sampler_rng'].cpu());flow.set_state(saved['flow_rng'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([a.cpu() for a in saved['cuda_rng']])
        np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng']);trace=saved['trace'];start=saved['update']+1
    t0=time.time();model.train()
    def save(step,path):
        tmp=path.with_name(path.name+'.tmp')
        torch.save({'update':step,'protocol_sha256':key,'model':model.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':sched.state_dict(),'sampler_rng':sampler.get_state(),'flow_rng':flow.get_state(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'trace':trace,'stats':stats},tmp)
        tmp.replace(path);path.with_suffix('.sha256').write_text(sha(path))
    for step in range(start,updates+1):
        idx=torch.multinomial(weights,batch_size,replacement=True,generator=sampler);x1=target[idx].to('cuda')
        sp=torch.full((batch_size,),24,dtype=torch.long,device='cuda');bbox=torch.zeros((batch_size,3),device='cuda');phy=torch.zeros((batch_size,32),device='cuda')
        _=torch.rand((),device='cuda',generator=flow).item()<.1
        t=torch.rand((batch_size,),device='cuda',generator=flow);x0=torch.randn(x1.shape,device='cuda',generator=flow)
        xt=(1-t[:,None])*x0+t[:,None]*x1
        pred=model(xt,t,sp,bbox,phy);fl=torch.nn.functional.mse_loss(pred,x1-x0)
        mid=(x0+x1)*.5;estimate=mid+.5*model(mid,torch.full((batch_size,),.5,device='cuda'),sp,bbox,phy)
        vl=torch.relu(x1.var()-estimate.var());loss=fl+.1*vl
        if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite flow loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step()
        if step%100==0 or step==updates:trace.append({'update':step,'loss':float(loss.detach()),'flow_loss':float(fl.detach()),'variance_loss':float(vl.detach())})
        if step%1000==0 or step==updates:
            save(step,resume);log(f'Flow seed{seed}: {step}/{updates}, elapsed {time.time()-t0:.1f}s')
    save(updates,final);model.eval();return model,stats,trace

def publish(paths,tag,folder_id=OUTPUT_FOLDER):
    """Write new immutable checkpoints/results only to the explicitly supplied experiment folder."""
    if not folder_id:return None
    from googleapiclient.http import MediaFileUpload
    dest=BASE/'exports';dest.mkdir(exist_ok=True);stamp=time.strftime('%Y%m%dT%H%M%S',time.gmtime())
    zpath=dest/f'{stamp}_{tag}.zip'
    files=[]
    for item in paths:
        p=Path(item)
        files.extend(q for q in p.rglob('*') if q.is_file()) if p.is_dir() else files.append(p)
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(set(files)):
            if p.exists() and p.suffix not in ('.npy',) and 'surface_samples' not in p.parts:z.write(p,p.relative_to(BASE))
    digest=sha(zpath)
    saved=_svc.files().create(body={'name':zpath.stem+'_'+digest[:12]+'.zip','parents':[folder_id],
        'description':'PhyloSDF tree-representation comparison; SHA256 '+digest},media_body=MediaFileUpload(str(zpath),mimetype='application/zip',resumable=True),fields='id,name,size,md5Checksum,webViewLink').execute()
    if int(saved['size'])!=zpath.stat().st_size or saved['md5Checksum']!=hashlib.md5(zpath.read_bytes()).hexdigest():raise RuntimeError('Drive upload verification failed')
    saved.update(sha256=digest,tag=tag)
    log('Saved '+saved['name']);return saved

def decode_resumable(rep,ckpt,rows,out,seed,folder_id):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);records=[];model,bench,parity,cksha=rep.decoder_for(ckpt)
    rep.exact_mode();t0=time.time();batch=[]
    for i,row in enumerate(rows,1):
        folder=out/'tasks'/row['task'];folder.mkdir(parents=True,exist_ok=True);rp=folder/'result.json'
        if rp.exists():
            record=json.loads(rp.read_text())
            if record['decoder_sha256']!=cksha:raise RuntimeError('Decode resume model changed')
            if record['status']=='completed' and sha(folder/'mesh.npz')!=record['output_sha256']['mesh.npz']:raise RuntimeError('Resume mesh hash mismatch')
        else:
            record={k:plain(v) for k,v in row.items() if k!='code'}
            record.update(group=PROTOCOL['schema'],historical_arm='beta_tree',paired_reference='beta_tree',training_seed=seed,
                generation_seed=seed,decoder='final.pt',decoder_sha256=cksha,resolution=256,zero_level=0,spacing=2/255,
                component_filtering='none',grid_frame='decoder normalized [-1,1]^3',finite_input_code=bool(np.isfinite(row['code']).all()))
            began=time.time()
            try:
                field=rep.sdf_field(model,row['code'],256);record.update(sdf_min=float(field.min()),sdf_max=float(field.max()))
                v,f,extractor=bench.extract(field,'skimage');record.update(bench.mesh_info(v,f))
                np.savez(folder/'mesh.npz',vertices=v,faces=f)
                record.update(status='completed',extractor=extractor,mesh_path=f'tasks/{row["task"]}/mesh.npz',output_sha256={'mesh.npz':sha(folder/'mesh.npz')})
            except Exception as e:record.update(status='mesh_failed',failure_type=type(e).__name__,failure=str(e))
            record['task_wall_seconds']=time.time()-began;write_json(rp,record);batch.append(folder)
        records.append(record)
        if i%20==0 or i==len(rows):
            write_json(out/'progress.json',{'done':i,'attempts':len(rows),'elapsed_seconds':time.time()-t0})
            log(f'{row["method"]} seed{seed}: decoded {i}/{len(rows)}, {(time.time()-t0)/i:.1f}s/mesh')
            if batch:publish([BASE/'PROTOCOL_LOCK.json',*batch],f'meshes_s{seed}_{row["method"]}_{i:03d}',folder_id);batch=[]
    write_json(out/'run_config.json',{'tasks':[{'task':r['task']} for r in records],'paired_reference':'beta_tree',
        'training_seed':seed,'decoder_sha256':cksha,'decoder_parity_max_abs':parity})
    write_json(out/'summary.json',{'schema':'tree_compare_decode_v1','records':records,'resolution':256,
        'valid_meshes':sum(r['status']=='completed' for r in records)})
    return out

def summarize():
    rows=[]
    diversity_path=BASE/'results/REFERENCE_DIVERSITY.json'
    reference_diversity=json.loads(diversity_path.read_text()) if diversity_path.exists() else {}
    for seed in SEEDS:
        for method in METHODS:
            p=BASE/f'results/seed{seed}_{method}/summary.json'
            if not p.exists():continue
            for g in json.loads(p.read_text()).get('groups',[]):
                geom=g.get('geometry_surface_seed_summary',{});get=lambda k:geom.get(k,{}).get('mean')
                mf=g.get('morphfid_decoder_frame_primary',{}).get('geometry8_no_count_or_centroid',{}).get('value')
                rows.append({'seed':seed,'method':method,'cohort':g['cohort'],'attempts':g['attempts'],'valid_meshes':g['valid_meshes'],
                    'nn_CD':get('generated_to_reference_nn_CD_mean'),'MMD_CD':get('MMD_CD_reference_to_generated'),
                    'COV':get('COV_reference_fraction'),'diversity_CD':get('diversity_all_pairs_CD_mean'),'MorphFID8':mf,
                    'within_species_diversity_percent':(100*get('diversity_all_pairs_CD_mean')/reference_diversity[g['cohort'][len('conditional_'):]]['mean_pairwise_CD']
                        if g['cohort'].startswith('conditional_') and get('diversity_all_pairs_CD_mean') is not None and g['cohort'][len('conditional_'):] in reference_diversity else None)})
    write_json(BASE/'results/COMPARISON_ROWS.json',rows)
    lines=['seed,method,cohort,attempts,valid_meshes,nn_CD,MMD_CD,COV,diversity_CD,MorphFID8,within_species_diversity_percent']
    lines += [','.join(str(r[k]) for k in ['seed','method','cohort','attempts','valid_meshes','nn_CD','MMD_CD','COV','diversity_CD','MorphFID8','within_species_diversity_percent']) for r in rows]
    (BASE/'results/COMPARISON_ROWS.csv').write_text('\n'.join(lines)+'\n')
    return rows

def preflight(svc,base=BASE,folder_id=OUTPUT_FOLDER):
    import torch,importlib.metadata
    rep=configure(base,svc);restore_archives(rep)
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required.')
    for package,required in [('trimesh','4.7.4'),('scikit-image','0.25.2')]:
        if importlib.metadata.version(package)!=required:raise RuntimeError('Install '+package+'=='+required+' before this run')
    key=lock_protocol(rep);ckpt,codes,species,checks=inputs(rep)
    prepare_references(rep)
    arrays,common=simple_samples(rep,codes,species,42)
    model,bench,parity,_=rep.decoder_for(ckpt);checks['decoder_numpy_torch_parity_max_abs']=parity
    checks['covariance_checks_passed']=True;checks['all_baseline_codes_finite']=True
    checks['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'python':platform.python_version(),'numpy':np.__version__,'trimesh':importlib.metadata.version('trimesh'),'scikit-image':importlib.metadata.version('scikit-image')}
    # A genuine256 grid extraction validates the actual decoder/extractor path; no quality claims.
    smoke=BASE/'preflight/smoke_mesh.npz';smoke.parent.mkdir(parents=True,exist_ok=True)
    start=time.time();field=rep.sdf_field(model,codes[0],256);v,f,_=bench.extract(field,'skimage');np.savez(smoke,vertices=v,faces=f)
    checks['genuine256_smoke']={'seconds':time.time()-start,'vertices':len(v),'faces':len(f),'sha256':sha(smoke)}
    vm=rep.vendor()
    refs,_=vm.load_references(rep.REFS/'reference_manifest.json')
    # Replay the original tree-reconstruction surface metric on the smoke mesh.
    original_rows=json.loads((BASE/'tree_results/results/beta_tree/metrics_reconstruction/reconstruction_per_specimen.json').read_text())
    archived=next(r for r in original_rows if r['specimen']==rep.specimens()[0][0])
    reference=next(r for r in refs if r['specimen']==archived['specimen'])
    replay=[]
    for surface_seed in rep.SURFACE_SEEDS:
        left=vm.sample_surface(v,f,10000,surface_seed)
        right=vm.sample_surface(reference['vertices'],reference['faces'],10000,surface_seed)
        replay.append(rep.kd_cd(left,right))
    replay_mean=float(np.mean(replay));expected=float(archived['mean_matched_specimen_CD'])
    checks['frozen_reconstruction_replay']={'specimen':archived['specimen'],'surface_seeds':list(rep.SURFACE_SEEDS),
        'CD_by_seed':replay,'CD_mean':replay_mean,'archived_CD_mean':expected,'abs_error':abs(replay_mean-expected)}
    if not np.isclose(replay_mean,expected,rtol=.001,atol=1e-7):raise RuntimeError('Frozen reconstruction metric replay failed: '+str(checks['frozen_reconstruction_replay']))
    # Exact raw-reference diversity for the four conditional cohorts, shared by all seeds/methods.
    reference_diversity={}
    for name in rep.FOCAL:
        selected=[r for r in refs if r['species']==name];per_seed=[]
        for surface_seed in rep.SURFACE_SEEDS:
            clouds=[vm.sample_surface(r['vertices'],r['faces'],10000,surface_seed) for r in selected]
            distances=[rep.kd_cd(clouds[i],clouds[j]) for i in range(len(clouds)) for j in range(i+1,len(clouds))]
            per_seed.append(float(np.mean(distances)))
        reference_diversity[name]={'pairwise_CD_by_surface_seed':per_seed,'mean_pairwise_CD':float(np.mean(per_seed)),
                                  'skulls':len(selected),'unordered_pairs':len(selected)*(len(selected)-1)//2}
    write_json(BASE/'results/REFERENCE_DIVERSITY.json',reference_diversity)
    checks['reference_count']=len(refs);checks['protocol_sha256']=key;checks['status']='passed'
    write_json(BASE/'results/PREFLIGHT.json',checks)
    source=BASE/'source';source.mkdir(exist_ok=True)
    shutil.copyfile(__file__,source/'tree_sampler_comparison.py');shutil.copyfile(Path(__file__).with_name('rep_helper.py'),source/'rep_helper.py')
    saved=publish([BASE/'PROTOCOL_LOCK.json',BASE/'results/PREFLIGHT.json',BASE/'results/REFERENCE_TRANSFORM_CHECKS.json',BASE/'results/REFERENCE_DIVERSITY.json',source,
        rep.REFS/'reference_manifest.json',rep.REFS/'original_reference_manifest.json',smoke],'preflight_and_source',folder_id)
    write_json(BASE/'results/PREFLIGHT_SAVED.json',saved);log('Preflight passed. No comparison metrics have yet been claimed.')
    return checks

def run(svc,base=BASE,folder_id=OUTPUT_FOLDER):
    import torch
    rep=configure(base,svc);lock_protocol(rep)
    if not (BASE/'results/PREFLIGHT.json').exists():raise RuntimeError('Run preflight first.')
    ckpt,codes,species,_=inputs(rep);vm=rep.vendor();vm.METHODS=tuple(vm.METHODS)+tuple(m for m in METHODS if m not in vm.METHODS)
    statuses=[]
    for seed in SEEDS:
        arrays,common=simple_samples(rep,codes,species,seed)
        # Complete all samplers for one seed before moving to the next independent seed.
        for method in METHODS:
            metrics=BASE/f'results/seed{seed}_{method}';marker=metrics/'ARM_COMPLETE.json'
            if marker.exists():
                recorded=json.loads(marker.read_text())
                if sha(metrics/'summary.json')!=recorded['summary_sha256']:raise RuntimeError('Completed metric summary checksum mismatch')
                if not (metrics/'DRIVE_SAVED.json').exists():
                    saved=publish([BASE/'PROTOCOL_LOCK.json',metrics],f'recovered_results_s{seed}_{method}',folder_id)
                    write_json(metrics/'DRIVE_SAVED.json',saved)
                statuses.append(recorded);continue
            gen=BASE/f'work/seed{seed}/{method}';gen.mkdir(parents=True,exist_ok=True);codepath=gen/'GENERATED_CODES.npz'
            if not codepath.exists():
                if method==METHODS[0]:
                    model,stats,trace=train_flow(rep,codes,species,seed)
                    target=rep.flow_sample(model,common['noise'])
                    generated=(rep.residuals_from_targets(target,stats)+stats['centroids'][common['species']]).astype(np.float32)
                    write_json(gen/'FLOW.json',{'seed':seed,'trace':trace,'generated_normalized_residual_variance':float(np.var(target,ddof=1)),
                        'stats':stats,'flow_checkpoint_sha256':sha(BASE/f'work/seed{seed}/flow/final.pt')})
                    del model;torch.cuda.empty_cache()
                else:generated=arrays[method]
                np.savez_compressed(codepath,codes=generated,**common)
                publish([BASE/'PROTOCOL_LOCK.json',gen,*([BASE/f'work/seed{seed}/flow'] if method==METHODS[0] else [])],f'generated_s{seed}_{method}',folder_id)
            data=np.load(codepath,allow_pickle=False)
            rows=[{'task':f'seed{seed}_{method}_{i:03d}','code':data['codes'][i],'method':method,'cohort':str(data['cohort'][i]),
                'sample_slot':int(data['sample_slot'][i]),'species':rep.species_names()[int(data['species'][i])],
                'species_idx':int(data['species'][i]),'source_array_row':i} for i in range(140)]
            decoded=decode_resumable(rep,ckpt,rows,gen/'decode',seed,folder_id)
            if not (metrics/'summary.json').exists():
                log(f'Evaluating all140 attempts: seed{seed} {method}')
                result=vm.run(rep.REFS/'reference_manifest.json',[decoded],metrics,points=10000,surface_seeds=rep.SURFACE_SEEDS,resume=True)
                rep.group_brief(result,f'seed{seed} {method}')
            status={'seed':seed,'method':method,'attempts':140,'status':'completed','summary_sha256':sha(metrics/'summary.json')}
            statuses.append(status);summarize()
            saved=publish([BASE/'PROTOCOL_LOCK.json',metrics,BASE/'results/COMPARISON_ROWS.json',BASE/'results/COMPARISON_ROWS.csv'],f'results_s{seed}_{method}',folder_id)
            write_json(metrics/'DRIVE_SAVED.json',saved)
            write_json(marker,status)
            write_json(BASE/'results/PROGRESS.json',{'completed_arms':len(statuses),'planned_arms':12,'statuses':statuses})
    write_json(BASE/'results/ALL_COMPLETE.json',{'status':'completed','arms':12,'attempts':1680,'statuses':statuses,
        'scope':PROTOCOL['scope']})
    publish([BASE/'PROTOCOL_LOCK.json',BASE/'results'],'all_results_complete',folder_id)
    log('All12 sampler/seed arms complete;1680 attempts. Read COMPARISON_ROWS.csv and full frozen summaries.')
    return summarize()
