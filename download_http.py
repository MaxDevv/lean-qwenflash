"""Parallel, resumable exact-range HF downloader."""
import argparse,os,json,time,urllib.request,concurrent.futures as cf,threading,mmap,hashlib
from pathlib import Path
from huggingface_hub import HfApi

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-dir', type=Path,
                    default=Path(os.environ.get('QWEN_WORK_ROOT','.'))/'model',
                    help='Checkpoint destination (default: ./model, or $QWEN_WORK_ROOT/model)')
# other uniform-bitrate quants too (e.g. 4.05bpw_h6_ng6), each pinned to one upload
parser.add_argument('--branch', default='3.05bpw_h5_ng5', help='Quant branch (default: 3.05bpw_h5_ng5)')
parser.add_argument('--sha', default=None, help='Revision to fetch (default: the pinned one for --branch)')
args=parser.parse_args()
root=args.model_dir.expanduser(); root.mkdir(parents=True,exist_ok=True)
base=root.parent
cg=Path(os.environ.get('QWEN_DOWNLOAD_CGROUP','/sys/fs/cgroup/qwen-download'))
if os.environ.get('QWEN_DOWNLOAD_JOIN_CGROUP','0')=='1':
 (cg/'cgroup.procs').write_text(str(os.getpid()))
# Each quant is fetched at a pinned revision, so a later upload to its branch can't change or mix an install.
# To move to a newer upload, test it and change the sha here (or pass --sha).
PINNED={'3.05bpw_h5_ng5':'69e33439ae950f17bcbe95c98f117d80f759ab6d',
        '4.05bpw_h6_ng6':'55a732e0c4c3d4614bc42b68493bb930d9b02c0a'}
def check(ok,msg):
 # explicit checks, not asserts: python -O would drop those
 if not ok: raise SystemExit(msg)
def digest(path,blob):
 # Hugging Face lists a sha256 for large (LFS) files and the git blob id (sha1 of a header + the bytes) for the rest
 h=hashlib.sha1(b'blob %d\0'%path.stat().st_size) if blob else hashlib.sha256()
 with open(path,'rb') as f:
  while b:=f.read(16*1024**2): h.update(b)
 return h.hexdigest()
def expected(s): return (s.lfs.sha256,False) if s.lfs else (s.blob_id,True)
repo='turboderp/Qwen3.8-Flash-Next-exl3'; branch=args.branch
rev=args.sha or PINNED.get(branch)
check(rev, f'no pinned revision for {branch}; pass --sha')
info=HfApi().model_info(repo,revision=rev,files_metadata=True)
print('REVISION',branch,info.sha,flush=True)
check(info.sha==rev, f'revision {info.sha} != expected {rev}')
# every file is checked against the hash Hugging Face lists for it; checked files are noted, so a rerun is quick
vpath=root/'.verified.json'
verified=json.loads(vpath.read_text()) if vpath.exists() else {}
def note(name,h):
 verified[name]=h; tmp=vpath.with_suffix('.tmp'); tmp.write_text(json.dumps(verified,indent=1)); tmp.replace(vpath)
chunk=64*1024**2; jobs=[]; files={}; lock=threading.Lock()
for s in info.siblings:
 rel=Path(s.rfilename)
 check(not rel.is_absolute() and '..' not in rel.parts, f'refusing a file outside the model folder: {s.rfilename!r}')
 dest=root/rel
 want,blob=expected(s)
 if dest.exists():
  if verified.get(s.rfilename)!=want:
   print('VERIFY',s.rfilename,flush=True)
   check(digest(dest,blob)==want, f'{dest} does not match the pinned upload; delete it and run this again')
   note(s.rfilename,want)
  continue
 dest.parent.mkdir(parents=True,exist_ok=True)
 size=s.size; u=f'https://huggingface.co/{repo}/resolve/{info.sha}/{s.rfilename}'
 with urllib.request.urlopen(urllib.request.Request(u,method='HEAD')) as r: url=r.geturl()
 part=root/(s.rfilename+'.http-part'); state=root/(s.rfilename+'.http-state.json')
 fd=os.open(part,os.O_CREAT|os.O_RDWR|os.O_DIRECT,0o644)
 os.ftruncate(fd,(size+4095)//4096*4096)
 os.posix_fallocate(fd,0,(size+4095)//4096*4096)
 done=set(json.loads(state.read_text())) if state.exists() else set()
 files[s.rfilename]={'size':size,'fd':fd,'part':part,'state':state,'done':done,'count':(size+chunk-1)//chunk,'sha256':s.lfs.sha256 if s.lfs else None,'want':want,'blob':blob}
 for i in range((size+chunk-1)//chunk):
  if i not in done: jobs.append((s.rfilename,url,i,i*chunk,min(size,(i+1)*chunk)-1))
def finish(name):
 """All chunks are in: trim the padding, check the hash, then move the file into place. False on a mismatch."""
 d=files[name]
 os.ftruncate(d['fd'],d['size']); os.fsync(d['fd']); os.close(d['fd'])
 if digest(d['part'],d['blob'])!=d['want']:
  d['part'].unlink(); d['state'].unlink(missing_ok=True)
  return False
 d['part'].replace(root/name); note(name,d['want']); print('COMPLETE',name,flush=True)
 return True
# a file whose chunks were all in when an earlier run stopped (e.g. during its hash check) is finished now
for name in [n for n,d in files.items() if len(d['done'])==d['count']]:
 check(finish(name), f'{name} does not match the pinned upload (deleted; run this again)')
# Round-robin files avoids leaving the giant table as a serial tail.
jobs.sort(key=lambda x:(x[2],x[0]))
t0=time.monotonic(); downloaded=0; local=threading.local()

def get(job):
 name,url,i,start,end=job; fd=files[name]['fd']
 for attempt in range(5):
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
   with urllib.request.urlopen(req,timeout=120) as r:
    if not (r.status==206 and r.headers.get('Content-Range')==f'bytes {start}-{end}/{files[name]["size"]}'): raise OSError('unexpected range response')
    pos=start
    if not hasattr(local,'buffer'): local.buffer=mmap.mmap(-1,1024**2)
    buf=memoryview(local.buffer)
    while pos<=end:
     need=min(len(buf),end+1-pos); got=0
     while got<need:
      n=r.readinto(buf[got:need])
      if not n: raise OSError('Short HTTP range response')
      got+=n
     padded=(need+4095)//4096*4096
     if padded>need: buf[need:padded]=bytes(padded-need)
     if os.pwrite(fd,buf[:padded],pos)!=padded: raise OSError('short write')
     pos+=need
    if pos!=end+1: raise OSError('incomplete range')
   return name,i,end-start+1
  except Exception:
   if attempt==4: raise
   time.sleep(1+attempt)

with cf.ThreadPoolExecutor(32) as ex:
 for j,future in enumerate(cf.as_completed([ex.submit(get,j) for j in jobs])):
  name,i,n=future.result(); downloaded+=n; d=files[name]; d['done'].add(i)
  if len(d['done'])%16==0 or len(d['done'])==d['count']:
   tmp=d['state'].with_suffix('.tmp'); tmp.write_text(json.dumps(sorted(d['done']))); tmp.replace(d['state'])
  if len(d['done'])==d['count'] and not finish(name):
   ex.shutdown(wait=False,cancel_futures=True)   # don't fetch the queued chunks before reporting
   raise SystemExit(f'{name} does not match the pinned upload (deleted; run this again)')
  if j%32==0: print('HTTP_PROGRESS',round(downloaded/1e9,2),'GB',round(time.monotonic()-t0,1),'s',round(downloaded/(time.monotonic()-t0)/1e9,3),'GB/s',flush=True)
result={'seconds':time.monotonic()-t0,'bytes':downloaded,'revision':info.sha,'files':{k:{'size':v['size'],'sha256':v['sha256']} for k,v in files.items()}}
(base/'http-download.json').write_text(json.dumps(result,indent=2))
with open(base/'restore.log','a') as f: f.write('\nDOWNLOAD_DONE\n')
print('DOWNLOAD_DONE',result['seconds'],flush=True)
