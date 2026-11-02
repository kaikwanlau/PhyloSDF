"""Fresh 79-code decoder: exact audited 2,000 epochs / 482,000 updates."""
from __future__ import annotations
from pathlib import Path
import copy, shutil, time
import numpy as np
from common import *


def prepare(study,reference_root):
    study=Path(study);root=Path(__file__).resolve().parent
    if study.exists():raise ValueError('Preparation requires a new dated study directory')
    expected={'deepsdf.py':'2526b8a129066179805a76930ee3286e597b9c6888d69d39b249061950245b51',
              'metadata.json':'7b1cd7f1a1ab1c989dbb69c48a6da03af1b5ada756f280901eca79766d459d05',
              'phylo_dist.npy':'5587ae53432b09162902425286c73183877c8faf687e14e73ac6a422878097f8'}
    for file,digest in expected.items():
        if sha(root/file)!=digest:raise ValueError('Frozen asset changed: '+file)
    meta=read_json(root/'metadata.json');refs=read_json(Path(reference_root)/'reference_manifest.json')
    rows={x['specimen']:x for x in refs['references']}
    if len(rows)!=100 or len(refs['references'])!=100:raise ValueError('Requires original complete100 reference manifest')
    train=[x for x in meta['specimens'] if x['species'] not in FOCAL];held=[x for x in meta['specimens'] if x['species'] in FOCAL]
    species=sorted({x['species'] for x in train})
    if (len(train),len(held),len(species),len({x['species'] for x in held}))!=(79,21,20,4):raise ValueError('Fixed split changed')
    if set(rows)!={x['filename'] for x in meta['specimens']}:raise ValueError('Reference specimen identity mismatch')
    for x in meta['specimens']:
        if rows[x['filename']]['species']!=x['species']:raise ValueError('Reference species mismatch')
    study.mkdir(parents=True)
    p={'schema':'group_holdout_full_v9','created_utc':utc(),'train':train,'heldout':held,'train_species':species,
       'heldout_species':list(FOCAL),'reference_manifest_sha256':sha(Path(reference_root)/'reference_manifest.json'),
       'source_sha256':source_hashes(),'architecture':ARCHITECTURE,
       'decoder_training':{'epochs':2000,'batch':16384,'drop_last':True,'rows_per_specimen':50000,'rows':3950000,
            'batches_per_epoch':241,'optimizer_updates':482000,'seed':42,'decoder_lr':.0005,'latent_lr':.001,
            'clamp':.1,'latent_l2':.001,'phylo_weight':.005,'phylo_batch_indices':'0,10,20,... each epoch',
            'scheduler_step_epochs':500,'scheduler_gamma':.5,'primary':'fixedfinal epoch2000',
            'best':'training-objective best only; sensitivity/recovery, never heldout selected',
            'permutation':'explicit torch.randperm CPU each epoch; persisted cursor and full permutation'},
       'latent_fitting':{'seeds':[42,43,44],'steps':2000,'batch':1024,'lr':.001,'clamp':.1,'latent_l2':.001,
            'init':'train79 global mean plus independent N(0,.01^2) seed-index jitter','selection':'fixedfinal2000',
            'fit_rows':40000,'diagnostic_rows':10000,'sealed_test_rows':50000,'split_seed':42},
       'generator':{'seeds':[42,43,44],'updates':10000,'lr':.0001,'weight_decay':.00001,'variance_weight':.1,
            'gradient_clip':1.,'scheduler':'cosine10000 eta_min1e-6','pool':'uniform79 specimen residuals',
            'conditions':'20 distances to training species only; scalar max train20x20 normalization',
            'centroid':'3nearest train species, inverse(proxy+1e-6), alphabetical ties',
            'samples_per_species':20,'generation_seed':2026101502,'euler':50,'CFG':1.,'temperature':1.},
       'evaluation':{'primary_resolution':128,'sensitivity_resolution':256,'grid':'endpointinclusive[-1,1]',
            'level':0.,'chunk':8192,'filtering':'none','surface_points':10000,'surface_seeds':[20261014,20261015],
            'MorphFID':'historical12, no-count11 and geometry8; train79 speciesbalanced zscore only',
            'reference_frame':'frozen per-specimen transform; not physicalsize',
            'scope':'retrospective4species group split; no prospective/24foldLOSO claim'},
       'lineage':'fresh decoder/codes/generators; no frozen100 or later101 weights',
       'phylo_caveat':'historical taxonomy proxy with sorted-key defect; no real branch lengths',
       'sdf_caveat':'historical sampled surface-point distances and3rayvote signs, not exact triangle signed distance',
       'heldout_geometry_opened_at_preparation':False,'input_rows':rows}
    atomic_json(study/'protocol.json',p)
    splits={}
    audited={x['filename']:x for x in read_json(root/'asset_audit.json')['specimens'] if x['role']=='heldout'}
    for i,row in enumerate(held):
        values=split_indices(filename=row['filename'])
        for key,prior in [('fit','fit'),('diagnostic','validation'),('test','test')]:
            if array_sha(values[key])!=audited[row['filename']]['proposed_point_subsplit']['parts'][prior]['indices_int64_le_sha256']:raise ValueError('Audited point split hash changed')
        for key,value in values.items():splits[row['filename']+'__'+key]=value
    save_npz(study/'SEALED_SPLIT_INDICES.npz',**splits)
    atomic_json(study/'SPLIT_SEAL.json',{'protocol_sha256':sha(study/'protocol.json'),'indices_sha256':sha(study/'SEALED_SPLIT_INDICES.npz'),
        'geometry_opened':False,'source_strata':STRATA,'utc':utc(),'chronology':'indices locked before fitting; enforcement is staged software, no third-party timestamp'})
    shutil.copyfile(root/'PROTOCOL.txt',study/'audited_protocol.txt')
    return p


def training_arrays(protocol,reference_root):
    np.random.seed(42);points=[];sdfs=[];ids=[];selected=[]
    for i,s in enumerate(protocol['train']):
        x,y=npz_data(reference_root,protocol['input_rows'][s['filename']]);sel=np.random.choice(100000,50000,replace=False)
        points.append(x[sel]);sdfs.append(y[sel]);ids.append(np.full(50000,i,np.int64));selected.append({'specimen':s['filename'],'sha256':array_sha(sel)})
    if len(points)!=79 or sum(len(v) for v in points)!=3950000:raise ValueError('Actual selected train rows differ')
    return np.concatenate(points),np.concatenate(sdfs),np.concatenate(ids),selected


def vector_phylo(codes,species,matrix):
    import torch
    ij=torch.triu_indices(len(codes),len(codes),1,device=codes.device)
    a=torch.linalg.vector_norm(codes[ij[0]]-codes[ij[1]],dim=1);b=matrix[species[ij[0]],species[ij[1]]]
    a=a-a.mean();b=b-b.mean()
    return -(a*b).sum()/(torch.sqrt(a.square().sum()*b.square().sum())+1e-8)


def equivalence(torch,library,labels,distances,out):
    from deepsdf import phylogenetic_consistency_loss
    sp=torch.tensor(labels,dtype=torch.long,device=library.latent_codes.weight.device)
    matrix=torch.tensor(distances,dtype=torch.float32,device=sp.device)
    a=phylogenetic_consistency_loss(library,labels,distances);ga=torch.autograd.grad(a,library.latent_codes.weight)[0]
    b=vector_phylo(library.latent_codes.weight,sp,matrix);gb=torch.autograd.grad(b,library.latent_codes.weight)[0]
    if not torch.allclose(a,b,atol=1e-6,rtol=1e-5) or not torch.allclose(ga,gb,atol=1e-5,rtol=1e-4):raise ValueError('Original/vector loss-gradient equivalence failed')
    atomic_json(out,{'passed':True,'actual_device':str(sp.device),'original':float(a.detach()),'vector':float(b.detach()),
        'gradient_max_abs_difference':float((ga-gb).abs().max()),'loss_atol':1e-6,'loss_rtol':1e-5,'gradient_atol':1e-5,'gradient_rtol':1e-4})
    return sp,matrix


def train(study,reference_root,archive_root,checkpoint_every=5000,stop_after=None,resume=False,device='cuda'):
    """stop_after bounds an invocation only; cannot shorten the fixed contract."""
    study=Path(study);out=study/'decoder';p=load_protocol(study,reference_root)
    if checkpoint_every<1:raise ValueError('Checkpoint cadence must be positive')
    if 79*ROWS_PER_SPECIMEN//BATCH!=BATCHES or EPOCHS*BATCHES!=UPDATES:raise ValueError('Budget algebra changed')
    bootstrap_archive(archive_root)
    torch=runtime(device,42)
    from deepsdf import DeepSDFDecoder,LatentCodeLibrary,deepsdf_loss
    with exclusive_writer(out):
        contract={'protocol_sha256':sha(study/'protocol.json'),'runtime':runtime_identity(torch,device),'checkpoint_every':checkpoint_every,
                  'stage':'fresh_exact2000epoch_decoder','budget':UPDATES}
        if (out/'checkpoint_manifest.json').exists() and not resume:raise ValueError('Existing stage requires --resume')
        x,y,idx,selection=training_arrays(p,reference_root)
        atomic_json(out/'TRAIN_ONLY_INPUTS.json',{'heldout_files_opened':[],'selection_hashes':selection,'rows':len(x),'train_count':79})
        x=torch.tensor(x,device=device);y=torch.tensor(y[:,None],device=device);idx=torch.tensor(idx,device=device)
        labels=np.array([p['train_species'].index(s['species']) for s in p['train']]);full=np.load(Path(__file__).parent/'phylo_dist.npy',allow_pickle=False)
        metadata=read_json(Path(__file__).parent/'metadata.json');species_ids=[metadata['species_to_idx'][s] for s in p['train_species']]
        distances=full[np.ix_(species_ids,species_ids)]
        decoder=DeepSDFDecoder(**ARCHITECTURE).to(device);library=LatentCodeLibrary(79,64).to(device)
        sp,matrix=equivalence(torch,library,labels,distances,out/'OBJECTIVE_EQUIVALENCE.json')
        od=torch.optim.Adam(decoder.parameters(),lr=.0005);ol=torch.optim.Adam(library.parameters(),lr=.001)
        sd=torch.optim.lr_scheduler.StepLR(od,500,.5);sl=torch.optim.lr_scheduler.StepLR(ol,500,.5)
        epoch=batch=updates=0;permutation=None;loss_sums=[0.,0.,0.];best_loss=None;best_state=None;epochs=[]
        if resume:
            state=restore(out,contract);decoder.load_state_dict(state['decoder']);library.load_state_dict(state['library'])
            od.load_state_dict(state['decoder_optimizer']);ol.load_state_dict(state['latent_optimizer']);sd.load_state_dict(state['decoder_scheduler']);sl.load_state_dict(state['latent_scheduler'])
            epoch,batch,updates=state['completed_epochs'],state['next_batch'],state['optimizer_updates'];permutation=state['permutation']
            loss_sums=state['loss_sums'];best_loss=state['best_loss'];best_state=state['best_state'];epochs=state['epoch_records']
            validate_cursor(epoch,batch,updates,permutation if permutation is not None else None);set_rng(torch,state,device)
        if stop_after is not None and stop_after<=updates and updates<UPDATES:raise ValueError('stop_after must exceed committed update')
        start=time.perf_counter();decoder.train()
        while updates<UPDATES:
            if permutation is None:permutation=torch.randperm(len(x),device='cpu')
            sel=permutation[batch*BATCH:(batch+1)*BATCH].to(device)
            codes=library(idx[sel]);prediction=decoder(codes,x[sel]);loss,recon,reg=deepsdf_loss(prediction,y[sel],codes,.1,.001)
            if batch%10==0:loss=loss+.005*vector_phylo(library.latent_codes.weight,sp,matrix)
            if not torch.isfinite(loss):raise ValueError('Nonfinite decoder objective')
            od.zero_grad();ol.zero_grad();loss.backward();od.step();ol.step();updates+=1
            loss_sums=[a+float(b.detach()) for a,b in zip(loss_sums,(loss,recon,reg))]
            epoch,batch,completed=next_cursor(epoch,batch)
            if completed:
                sd.step();sl.step();record={'epoch':epoch,'optimizer_updates':updates,'loss':loss_sums[0]/BATCHES,
                    'reconstruction':loss_sums[1]/BATCHES,'latent_mse':loss_sums[2]/BATCHES,'decoder_lr':od.param_groups[0]['lr'],'latent_lr':ol.param_groups[0]['lr']}
                epochs.append(record)
                if best_loss is None or record['loss']<best_loss:
                    best_loss=record['loss'];best_state={'decoder':{k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()},
                         'library':{k:v.detach().cpu().clone() for k,v in library.state_dict().items()},'epoch':epoch,'loss':best_loss}
                loss_sums=[0.,0.,0.];permutation=None
                if epoch%50==0 or epoch==1:print(record,flush=True)
            validate_cursor(epoch,batch,updates)
            due=updates%checkpoint_every==0 or updates==UPDATES or (stop_after is not None and updates>=stop_after)
            if due:
                state={'contract':contract,'optimizer_updates':updates,'completed_epochs':epoch,'next_batch':batch,'permutation':permutation,
                    'decoder':decoder.state_dict(),'library':library.state_dict(),'decoder_optimizer':od.state_dict(),'latent_optimizer':ol.state_dict(),
                    'decoder_scheduler':sd.state_dict(),'latent_scheduler':sl.state_dict(),'loss_sums':loss_sums,'best_loss':best_loss,
                    'best_state':best_state,'epoch_records':epochs,**rng_state(torch)}
                checkpoint(out,state,archive_root)
                atomic_json(out/'epoch_records.json',epochs)
                atomic_json(out/'STATUS.json',{'status':'complete' if updates==UPDATES else 'running_or_paused','updates':updates,'epochs':epoch,'next_batch':batch,
                    'heldout_files_opened':[],'elapsed_invocation_seconds':time.perf_counter()-start,'archive_readback_completed':True})
                print({'committed_updates':updates,'epochs':epoch,'batch':batch},flush=True)
                if stop_after is not None and updates>=stop_after:break
        if updates==UPDATES:
            if epoch!=2000 or batch!=0 or sd.last_epoch!=2000 or sl.last_epoch!=2000:raise ValueError('Final scheduler/budget identity failed')
            # Fixed final is immutable; no best selection enters downstream primary.
            final={'contract':contract,'optimizer_updates':updates,'completed_epochs':epoch,'decoder':decoder.state_dict(),'library':library.state_dict()}
            path=out/'final.pt';tmp=out/'final.pt.tmp'
            with tmp.open('wb') as f:torch.save(final,f)
            tmp.replace(path);archive_file(path,out,archive_root,final)
            best_path=out/'best_sensitivity.pt'
            with best_path.open('wb') as f:torch.save(best_state,f)
            archive_file(best_path,out,archive_root,best_state)
            save_npz(out/'train_codes.npz',codes=library.get_all_codes(),species_labels=labels,train_species=np.asarray(p['train_species']))
            atomic_json(out/'COMPLETION.json',{'status':'fixed_final_completed','epochs':2000,'updates':482000,'final_sha256':sha(path),
                'train_codes_sha256':sha(out/'train_codes.npz'),'heldout_files_opened':[],'primary':'final.pt','best_sensitivity_sha256':sha(best_path)})
            ensure_archived(out/'COMPLETION.json',out,archive_root)
        return updates
