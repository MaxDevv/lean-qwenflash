"""Natural-completion C1 benchmark; results are persisted after each request."""
import argparse, json, os, time, threading, subprocess
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--store',required=True)
    ap.add_argument('--chunk',type=int,default=2048); ap.add_argument('--ram-gb',type=float,default=6)
    ap.add_argument('--cache',type=int,default=16384); ap.add_argument('--out',required=True)
    ap.add_argument('--prefill',type=int,nargs='*',default=[1024,8192])
    ap.add_argument('--reps',type=int,default=2); ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--mtp',type=int,default=0); ap.add_argument('--mtp-grid',type=int,nargs='*',default=[])
    ap.add_argument('--prefill-file',default=None)
    ap.add_argument('--decode-json',default=None)
    ap.add_argument('--prefill-tail',default='Now write exactly one sentence explaining why the sky is blue.')
    ap.add_argument('--retrieval',type=int,nargs='*',default=[])
    ap.add_argument('--prefetch-compare',action='store_true')
    ap.add_argument('--single-pass-prefill',action='store_true')
    ap.add_argument('--stage-min-compare',type=int,nargs='*',default=[])
    ap.add_argument('--copy-grid-compare',type=int,nargs='*',default=[])
    ap.add_argument('--qsa-tile-compare',type=int,nargs='*',default=[])
    ap.add_argument('--compare-context',type=int,default=100000)
    ap.add_argument('--embed-disk',action='store_true')
    ap.add_argument('--cache-quant',default=None)
    ap.add_argument('--verify',action='store_true')
    ap.add_argument('--decode-profile',action='store_true',help='Snapshot GPU wait counters at first token for decode-only deltas')
    ap.add_argument('--cpu-compare',action='store_true')
    ap.add_argument('--cpu-compare-reps',type=int,default=2)
    a=ap.parse_args()
    if a.single_pass_prefill: os.environ['QWEN_EXL3_SOURCE']='1'
    from memory_guard import DeviceMemory,HostMemory
    gpu_memory=DeviceMemory()
    host_memory=HostMemory(os.environ.get('QWEN_GLOBAL_MEMORY_CGROUP',
                                         os.environ.get('QWEN_MEMORY_CGROUP','/sys/fs/cgroup')))
    from qwen_offload import load
    model,config,cache,tok,pools,tables,draft_model,draft_cache=load(
        a.model,a.store,a.chunk,a.cache,a.ram_gb,a.cpu,max([a.mtp]+a.mtp_grid),a.embed_disk,a.cache_quant)
    import torch
    from exllamav3 import Generator,Job,GreedySampler
    gen=None
    eos=tok.single_id('<|im_end|>')
    result={'args':dict(vars(a)),'runs':[],'started':time.time(),'env':{k:v for k,v in os.environ.items() if k.startswith(('GLM53_','EXL3_','QWEN_'))}}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    # Exercise actual disk gathers and independently implemented hashing/trellis decoding.
    result['ngram_validation']=[]
    with torch.inference_mode():
        for table in tables:
            ids=torch.randint(0,50000,(1,66),generator=torch.Generator().manual_seed(123))
            ids[0,:2]=table.eos_token_id; ids[0,32]=table.eos_token_id
            fast=table.forward(ids,{})
            reference=table.forward_reference(ids,{})
            delta=(fast.float()-reference.float())
            check={'key':table.key,'mode':table.mode,'equal':torch.equal(fast,reference),
                   'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/reference.float().norm())}
            result['ngram_validation'].append(check)
            assert check['relative_l2']<0.001,check
    def save():
        tmp=out.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2)); tmp.replace(out)
    def mem():
        cg=Path(os.environ.get('QWEN_MEMORY_CGROUP','/sys/fs/cgroup'))
        return {'memory.current':int((cg/'memory.current').read_text()),
                'memory.sampled_peak':host_memory.peak,
                'memory.kernel_historical_peak':int((cg/'memory.peak').read_text())}
    def run(instruction,size=None,variant='base',retrieval=False):
        nonlocal gen
        # Fresh generator: no prefix reuse between requests; expert tiers remain warm.
        gen=Generator(model,cache,tok,max_batch_size=1,max_chunk_size=a.chunk,
                      recurrent_cache_size=256*1024**2,draft_model=draft_model if a.mtp else None,
                      draft_cache=draft_cache if a.mtp else None,num_draft_tokens=a.mtp if a.mtp else None,
                      record_draft_stats=bool(a.mtp))
        gen.single_pass_prefill=a.single_pass_prefill
        # Distinct text per repetition prevents accidental prefix-cache benchmarks.
        prompt='<|im_start|>user\n'+instruction
        if size:
            if a.prefill_file:
                source=Path(a.prefill_file).read_text()
                filler=source*(1+size*16//max(len(source),1))
            else:
                filler=('The fox observes the river, while the engineer records temperature and pressure. ' * size)
            ids=tok.encode(prompt+'\n'+filler,encode_special_tokens=True)[:,:size-40]
            tail_text=a.prefill_tail
            if retrieval:
                ids=ids.clone()
                for fraction,name,value in ((0.1,'ARCTIC','739164'),(0.5,'DESERT','285903'),(0.9,'FOREST','614827')):
                    entry=tok.encode(f'\nThe secret access code for {name} is {value}.\n',encode_special_tokens=True)
                    position=int(ids.shape[1]*fraction)
                    ids[:,position:position+entry.shape[1]]=entry
                tail_text='What are the secret access codes for ARCTIC, DESERT, and FOREST? Reply only with the three codes in that order.'
            tail=tok.encode('\n'+tail_text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',encode_special_tokens=True)
            ids=torch.cat((ids,tail),1)
        else:
            ids=tok.encode(prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',encode_special_tokens=True)
        job=Job(input_ids=ids,sampler=GreedySampler(),stop_conditions=[eos])
        nv_before=[p.nv2.counters() for p in pools]
        gen.enqueue(job); text=''; steps=[]; iterations=[]; first=None; nv_first=None; nv_device_first=None; t0=time.perf_counter()
        while gen.num_remaining_jobs():
            before_tokens=job.new_tokens
            t=time.perf_counter(); events=gen.iterate(); dt=time.perf_counter()-t
            iterations.append({'seconds':dt,'tokens':job.new_tokens-before_tokens})
            for e in events:
                if e.get('stage')=='error':
                    result.setdefault('errors',[]).append({'variant':variant,'error':repr(e['error'])})
                    save()
                    raise e['error']
                if e.get('stage')=='streaming':
                    if first is None:
                        first=time.perf_counter(); nv_first=[p.nv2.counters() for p in pools]
                        if a.decode_profile:
                            nv_device_first=[p.nv2.summary()['dev'] for p in pools]
                    steps.append(dt); text+=e.get('text','')
                if e.get('eos'): final=e
        torch.cuda.synchronize()
        assert job.new_tokens>0 and first is not None, 'Generator produced no tokens; not a valid benchmark'
        row={'variant':variant,'chunk':a.chunk,'mtp':a.mtp,
             'prefetch':pools[0].nv2.prefetch,'stage_min':pools[0].stage_min,
             'single_pass_prefill':a.single_pass_prefill,
             'qsa_score_tile':__import__('exllamav3.modules.qsa_indexer',fromlist=['QSAIndexer']).QSAIndexer.SEL_TILE,
             'selective_prefill':os.environ.get('QWEN_SELECTIVE_PREFILL')=='1',
             'prompt_tokens':ids.shape[1],'output':text,'wall_seconds':time.perf_counter()-t0,
             'tokens':job.new_tokens,'prefill_seconds':job.time_prefill,'decode_seconds':job.time_generate,
             'draft_accepted':job.accepted_draft_tokens,'draft_rejected':job.rejected_draft_tokens,
             'prefill_tps':ids.shape[1]/max(job.time_prefill,1e-9),
             'decode_tps':max(job.new_tokens-1,0)/max(job.time_generate,1e-9),
             'eos_reason':final.get('eos_reason'),'step_seconds':steps,'iterations':iterations,'memory':mem(),
             'nv2_before':nv_before,'nv2_first_stream':nv_first,
             'nv2_summary':[p.nv2.summary() for p in pools],
             'total_vram_peak':gpu_memory.peak,'global_ram_sampled_peak':host_memory.peak,
             'global_ram_current':host_memory.sample(),
             'global_ram_cap':(host_memory.path/'memory.max').read_text().strip(),
             'cuda_peak_allocated':torch.cuda.max_memory_allocated(),
             'cuda_peak_reserved':torch.cuda.max_memory_reserved(),
             'tiers':[p.summary() for p in pools], 'nv2':[p.nv2.counters() for p in pools],
             'ngram':[m.prefetch_stats for m in tables]}
        if a.decode_profile: row['nv2_device_first_stream']=nv_device_first
        if retrieval:
            row['retrieval']={'depths':[0.1,0.5,0.9],'expected':['739164','285903','614827'],
                              'passed':all(code in text for code in ('739164','285903','614827'))}
        if a.verify: row['verification']=[p.nv2.verify(16) for p in pools]
        result['runs'].append(row); save()
        # The last native request remains protected until the next publication;
        # identical GPU/RAM copies at this boundary are diagnostic, not corrupt data.
        for v in row.get('verification',[]):
            assert all(v[k]==0 for k in ('vram_bad','ram_bad','table_bad','home_bad_resident',
                                         'home_bad_nonresident','dev_bad_rows')),v
        print('RESULT',json.dumps({k:v for k,v in row.items() if k not in
              ('tiers','nv2','nv2_before','nv2_first_stream','nv2_summary','step_seconds','iterations')}),flush=True)
    save()
    run('Compute 17 times 23. Reply with only the number.')
    for r in range(a.reps):
        run(f'Trial {r}: Explain how a jet engine works in three concise paragraphs.')
        for size in a.prefill: run(f'Trial {r}, document {size}: Read the following document.',size)
    for size in a.retrieval:
        run('Read the following document carefully and retain its access codes.',size,
            variant=f'retrieval-{size}',retrieval=True)
    for tile in a.qsa_tile_compare:
        from exllamav3.modules.qsa_indexer import QSAIndexer
        QSAIndexer.SEL_TILE=tile
        run('Read the following document.',a.compare_context,variant=f'qsa-tile-{tile}')
    for threshold in a.stage_min_compare:
        for p in pools: p.stage_min=threshold
        run('Trial 0: Explain how a jet engine works in three concise paragraphs.',variant=f'stage-min-{threshold}')
    for grid in a.copy_grid_compare:
        for p in pools: p.nv2.grid=grid
        run('Trial 0: Explain how a jet engine works in three concise paragraphs.',variant=f'copy-grid-{grid}')
    if a.prefetch_compare:
        for on in (False,True,True,False):
            for p in pools:
                p.nv2.prefetch=on
                p.nv2._pfe=None
            run('Trial 0: Explain how a jet engine works in three concise paragraphs.',variant=f'prefetch-{int(on)}')
    for mtp in a.mtp_grid:
        a.mtp=mtp
        instructions=json.loads(Path(a.decode_json).read_text()) if a.decode_json else [
            'Trial 0: Explain how a jet engine works in three concise paragraphs.']
        for index,instruction in enumerate(instructions):
            run(instruction,variant=f'mtp-{mtp}-prompt-{index}')
    if a.cpu_compare:
        for p in pools: p.nv2.enable_cpu()
        for rep in range(a.cpu_compare_reps):
            for on in ((False,True) if rep%2==0 else (True,False)):
                for p in pools: p.nv2.set_cpu(on,nvcpu=on)
                run('Trial 0: Explain how a jet engine works in three concise paragraphs.',
                    variant=f'cpu-{int(on)}-rep-{rep}')
    save()

if __name__=='__main__': main()
