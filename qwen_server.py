"""Qwen adapter for the existing OpenAI-compatible offload server."""
import argparse,importlib.util,json,os,re,time,uuid
from pathlib import Path
from types import SimpleNamespace
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT=Path(__file__).resolve().parent
TOOL_CALL=re.compile(r'<tool_call>(.*?)(?:</tool_call>|$)',re.S)
FUNCTION=re.compile(r'<function=([^>]+)>(.*?)</function>',re.S)
PARAMETER=re.compile(r'<parameter=([^>]+)>(.*?)</parameter>',re.S)

def parse_qwen_tools(content,tools):
    if not tools or '<tool_call>' not in content:return content,[]
    schemas={t.get('function',t)['name']:t.get('function',t).get('parameters',{}).get('properties',{}) for t in tools}
    calls=[]
    for block in TOOL_CALL.findall(content):
        for name,body in FUNCTION.findall(block):
            name=name.strip();arguments={}
            for key,value in PARAMETER.findall(body):
                key=key.strip();value=value.strip()
                try:parsed=json.loads(value)
                except ValueError:parsed=value
                if schemas.get(name,{}).get(key,{}).get('type')=='string' and not isinstance(parsed,str):parsed=value
                arguments[key]=parsed
            calls.append({'id':'call_'+uuid.uuid4().hex[:24],'type':'function',
                          'function':{'name':name,'arguments':json.dumps(arguments,ensure_ascii=False)}})
    return (TOOL_CALL.sub('',content).strip(),calls) if calls else (content,[])

def qwen_sampling_defaults(body):
    """Model-card defaults; explicit client sampling values still take precedence."""
    thinking=(body.get('chat_template_kwargs') or {}).get('enable_thinking',True)
    defaults={'temperature':1.0 if thinking else 0.7,
              'top_p':0.95 if thinking else 0.80,'top_k':20,'min_p':0.0}
    return {**defaults,**{k:v for k,v in body.items() if v is not None}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default=str(ROOT/'model'));ap.add_argument('--store',default=str(ROOT/'store/qwen.bin'))
    ap.add_argument('--ram-gb',type=float,required=True);ap.add_argument('--cache',type=int,default=262144)
    ap.add_argument('--chunk',type=int,default=24576);ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--port',type=int,default=30000);ap.add_argument('--served-name',default='qwen3.8-flash-next')
    cli=ap.parse_args()
    os.environ['QWEN_EXL3_SOURCE']='1'
    from memory_guard import DeviceMemory,HostMemory
    gpu=DeviceMemory();host=HostMemory(os.environ['QWEN_GLOBAL_MEMORY_CGROUP'])
    from qwen_offload import load
    start=time.monotonic()
    model,config,cache,tok,pools,tables,draft_model,draft_cache=load(
        cli.model,cli.store,cli.chunk,cli.cache,cli.ram_gb,cpu=False,mtp=0,embed_disk=True,cache_quant=4)
    # Reuse the existing HTTP, SSE, disconnect cancellation, sampling and stats implementation.
    spec=importlib.util.spec_from_file_location('qwen_http_backend',ROOT/'qwen38-flash-offload/qwen38/serve.py')
    backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
    backend.args=SimpleNamespace(model_dir=cli.model,cache_size=cli.cache,chunk_size=cli.chunk,cache_quant='4',
        host=cli.host,port=cli.port,served_name=cli.served_name,max_batch_size=1,max_q_size=8,
        num_draft_tokens=0,dynamic_draft=False,draft_confidence=0.0,ngram_match_min=0,recurrent_cache_size=0.25,
        expert_ram_gib=cli.ram_gb,single_pass_prefill=False)
    backend.model=model;backend.config=config;backend.cache=cache;backend.tok=tok
    backend.draft_model=draft_model;backend.draft_cache=draft_cache;backend.CTX=cli.cache
    backend.t_load=time.monotonic()-start;backend.hf_tok=None
    backend.EOS=sorted(set((getattr(config,'eos_token_id_list',None) or [])+
                          [tok.single_id('<|im_end|>')]+([tok.eos_token_id] if tok.eos_token_id is not None else [])))
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    def fail(message):raise ValueError(message)
    env.globals['raise_exception']=fail
    env.filters['tojson']=lambda value:json.dumps(value,ensure_ascii=False)
    template=env.from_string((Path(cli.model)/'chat_template.jinja').read_text())
    def chat_prompt(messages,enable_thinking=True,tools=None,template_kwargs=None):
        kw=dict(template_kwargs or {});kw.pop('enable_thinking',None)
        return template.render(messages=backend.normalize_messages(messages),tools=tools,
                               add_generation_prompt=True,enable_thinking=enable_thinking,**kw)
    backend.chat_prompt=chat_prompt;backend.parse_tool_calls=parse_qwen_tools
    original_openai_sp=backend.openai_sp
    backend.openai_sp=lambda body:original_openai_sp(qwen_sampling_defaults(body))
    class ObservedJob(backend.AsyncJob):
        def __init__(self,*args,**kwargs):
            self.submitted=time.monotonic()
            super().__init__(*args,**kwargs)
        async def __aiter__(self):
            from exllamav3.constants import PAGE_SIZE
            first=True
            async for result in super().__aiter__():
                if first and result.get('stage')=='streaming':
                    first=False
                    rc=self.generator.generator.recurrent_cache
                    metric={'prompt_tokens':len(self.job.sequences[0].sequence_ids),
                            'cached_tokens':self.job.cached_pages*PAGE_SIZE+self.job.cached_tokens,
                            'first_stream_seconds':time.monotonic()-self.submitted,
                            'recurrent_checkpoints':len(rc) if rc is not None else 0,
                            'recurrent_cache_bytes':rc.current_size if rc is not None else 0}
                    backend.G['stats']['last_request_prefill']=metric
                    print('QWEN_PREFILL',json.dumps(metric),flush=True)
                yield result
    backend.AsyncJob=ObservedJob
    original_startup=backend.startup
    async def startup():
        await original_startup()
        # Serving reuses prefixes: retain the final page-aligned recurrent checkpoint.
        # Single-pass prefill is only appropriate for fresh-generator benchmark requests.
        backend.G['gen'].generator.single_pass_prefill=False
    # Replace the existing startup callback, retaining the same lifecycle and endpoints.
    backend.app.router.on_startup[:]=[startup]
    @backend.app.on_event('shutdown')
    async def shutdown():
        if backend.G['gen'] is not None:await backend.G['gen'].close()
        gpu.close();host.close()
    original_run_job=backend.run_job
    async def run_job(ids,sp,stop_strings=None):
        if ids.shape[-1]>=cli.cache-10:
            from fastapi import HTTPException
            raise HTTPException(400,'Prompt leaves no generation room in the configured context.')
        async for event in original_run_job(ids,sp,stop_strings):yield event
    backend.run_job=run_job
    @backend.app.get('/memory')
    async def memory():
        return {'ram_current':host.sample(),'ram_sampled_peak':host.peak,'ram_limit':host.limit,
                'total_vram_current':gpu.sample(),'total_vram_peak':gpu.peak,'vram_limit':gpu.limit}
    print('QWEN_SERVER_LOADED',backend.t_load,'seconds','RAM_CACHE_GiB',cli.ram_gb,'CONTEXT',cli.cache,flush=True)
    import uvicorn
    uvicorn.run(backend.app,host=cli.host,port=cli.port,log_level='info')

if __name__=='__main__':main()
