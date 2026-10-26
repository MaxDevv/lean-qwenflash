"""Parallel, resumable exact-range HF downloader."""
import argparse,os,json,time,urllib.request,concurrent.futures as cf,threading,mmap
from pathlib import Path
from huggingface_hub import HfApi

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-dir', type=Path,
                    default=Path(os.environ.get('QWEN_WORK_ROOT','.'))/'model',
                    help='Checkpoint destination (default: ./model, or $QWEN_WORK_ROOT/model)')
args=parser.parse_args()
root=args.model_dir.expanduser(); root.mkdir(parents=True,exist_ok=True)
base=root.parent
cg=Path(os.environ.get('QWEN_DOWNLOAD_CGROUP','/sys/fs/cgroup/qwen-download'))
if os.environ.get('QWEN_DOWNLOAD_JOIN_CGROUP','0')=='1':
 (cg/'cgroup.procs').write_text(str(os.getpid()))
repo='turboderp/Qwen3.8-Flash-Next-exl3'; branch='3.05bpw_h5_ng5'
info=HfApi().model_info(repo,revision=branch,files_metadata=True)
assert info.sha=='69e33439ae950f17bcbe95c98f117d80f759ab6d'
chunk=64*1024**2; jobs=[]; files={}; lock=threading.Lock()
for s in info.siblings:
 if (root/s.rfilename).exists(): continue
 size=s.size; u=f'https://huggingface.co/{repo}/resolve/{branch}/{s.rfilename}'
 with urllib.request.urlopen(urllib.request.Request(u,method='HEAD')) as r: url=r.geturl()
 part=root/(s.rfilename+'.http-part'); state=root/(s.rfilename+'.http-state.json')
 fd=os.open(part,os.O_CREAT|os.O_RDWR|os.O_DIRECT,0o644)
 os.ftruncate(fd,(size+4095)//4096*4096)
 os.posix_fallocate(fd,0,(size+4095)//4096*4096)
 done=set(json.loads(state.read_text())) if state.exists() else set()
 files[s.rfilename]={'size':size,'fd':fd,'part':part,'state':state,'done':done,'count':(size+chunk-1)//chunk,'sha256':s.lfs.sha256 if s.lfs else None}
 for i in range((size+chunk-1)//chunk):
  if i not in done: jobs.append((s.rfilename,url,i,i*chunk,min(size,(i+1)*chunk)-1))
# Round-robin files avoids leaving the giant table as a serial tail.
jobs.sort(key=lambda x:(x[2],x[0]))
t0=time.monotonic(); downloaded=0; local=threading.local()

def get(job):
 name,url,i,start,end=job; fd=files[name]['fd']
 for attempt in range(5):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=120) as r:
    assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{files[name]["size"]}'
    pos=start
    if not hasattr(local,'buffer'): local.buffer=mmap.mmap(-1,1024**2)
    buf=memoryview(local.buffer)
    while pos<=end:
     need=min(len(buf),end+1-pos); got=0
     while got<need:
      n=r.readinto(buf[got:need]); assert n>0, 'Short HTTP range response'; got+=n
     padded=(need+4095)//4096*4096
     if padded>need: buf[need:padded]=bytes(padded-need)
     assert os.pwrite(fd,buf[:padded],pos)==padded
     pos+=need
    assert pos==end+1
   return name,i,end-start+1
  except Exception:
   if attempt==4: raise
   time.sleep(1+attempt)

with cf.ThreadPoolExecutor(32) as ex:
 for j,future in enumerate(cf.as_completed([ex.submit(get,j) for j in jobs])):
  name,i,n=future.result(); downloaded+=n; d=files[name]; d['done'].add(i)
  if len(d['done'])%16==0 or len(d['done'])==d['count']:
   tmp=d['state'].with_suffix('.tmp'); tmp.write_text(json.dumps(sorted(d['done']))); tmp.replace(d['state'])
  if len(d['done'])==d['count']:
   os.ftruncate(d['fd'],d['size']); os.fsync(d['fd']); os.close(d['fd'])
   d['part'].replace(root/name); print('COMPLETE',name,flush=True)
  if j%32==0: print('HTTP_PROGRESS',round(downloaded/1e9,2),'GB',round(time.monotonic()-t0,1),'s',round(downloaded/(time.monotonic()-t0)/1e9,3),'GB/s',flush=True)
result={'seconds':time.monotonic()-t0,'bytes':downloaded,'revision':info.sha,'files':{k:{'size':v['size'],'sha256':v['sha256']} for k,v in files.items()}}
(base/'http-download.json').write_text(json.dumps(result,indent=2))
with open(base/'restore.log','a') as f: f.write('\nDOWNLOAD_DONE\n')
print('DOWNLOAD_DONE',result['seconds'],flush=True)
