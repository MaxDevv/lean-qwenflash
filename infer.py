"""Single-stream text inference using the capped Qwen offload engine."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True); ap.add_argument('--store',required=True)
    ap.add_argument('--prompt'); ap.add_argument('--raw',action='store_true')
    ap.add_argument('--no-thinking',action='store_true')
    ap.add_argument('--chunk',type=int,default=8192); ap.add_argument('--cache',type=int,default=16384)
    ap.add_argument('--single-pass-prefill',action='store_true')
    ap.add_argument('--ram-gb',type=float,default=9); ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--mtp',type=int,default=0); ap.add_argument('--cache-quant',default=None)
    ap.add_argument('--metrics',help='Persist request metrics to this JSON file')
    a=ap.parse_args()
    from memory_guard import HostMemory,DeviceMemory
    host=HostMemory(os.environ.get('QWEN_GLOBAL_MEMORY_CGROUP',os.environ.get('QWEN_MEMORY_CGROUP','/sys/fs/cgroup')))
    gpu=DeviceMemory()
    if a.single_pass_prefill: os.environ['QWEN_EXL3_SOURCE']='1'
    from qwen_offload import load
    model,config,cache,tok,pools,tables,draft,draft_cache=load(
        a.model,a.store,a.chunk,a.cache,a.ram_gb,a.cpu,a.mtp,True,a.cache_quant)
    from exllamav3 import Generator,Job,GreedySampler
    prompt=a.prompt if a.prompt is not None else sys.stdin.read()
    rendered=prompt if a.raw else model.default_chat_prompt(prompt)
    if a.no_thinking and not a.raw: rendered+='<think>\n\n</think>\n\n'
    ids=tok.encode(rendered,encode_special_tokens=True)
    gen=Generator(model,cache,tok,max_batch_size=1,max_chunk_size=a.chunk,
                  recurrent_cache_size=256*1024**2,draft_model=draft,draft_cache=draft_cache,
                  num_draft_tokens=a.mtp if a.mtp else None,record_draft_stats=bool(a.mtp))
    gen.single_pass_prefill=a.single_pass_prefill
    job=Job(input_ids=ids,sampler=GreedySampler(),stop_conditions=[tok.single_id('<|im_end|>')])
    gen.enqueue(job); start=time.perf_counter(); text=''
    while gen.num_remaining_jobs():
        for event in gen.iterate():
            if event.get('stage')=='error': raise event['error']
            if event.get('stage')=='streaming':
                part=event.get('text',''); text+=part; print(part,end='',flush=True)
    print()
    metrics={'prompt_tokens':ids.shape[1],'tokens':job.new_tokens,'output':text,
             'wall_seconds':time.perf_counter()-start,'prefill_seconds':job.time_prefill,
             'decode_seconds':job.time_generate,'decode_tps':max(job.new_tokens-1,0)/max(job.time_generate,1e-9),
             'draft_accepted':job.accepted_draft_tokens,'draft_rejected':job.rejected_draft_tokens,
             'global_ram_sampled_peak':host.peak,'total_vram_peak':gpu.peak}
    if a.metrics:
        p=Path(a.metrics); p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(metrics,indent=2))
    print(json.dumps({k:v for k,v in metrics.items() if k!='output'}),file=sys.stderr)

if __name__=='__main__': main()
