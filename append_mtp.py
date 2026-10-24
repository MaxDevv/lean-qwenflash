"""Append the selected quant's MTP experts without repacking the trunk's 45.8 GB."""
import argparse,concurrent.futures as cf,hashlib,importlib.util,json,mmap,os,threading,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--store',required=True); a=ap.parse_args()
    p=Path(__file__).resolve().parent/'glm53-flash-offload/scripts/pack_glm53_store.py'
    spec=importlib.util.spec_from_file_location('packer',p); packer=importlib.util.module_from_spec(spec); spec.loader.exec_module(packer)
    store=Path(a.store); manifest=store.with_suffix('.json'); meta=json.loads(manifest.read_text())
    if meta.get('mtp_layer_li') is not None: print('MTP already packed'); return
    src=packer.Src(a.model,True)
    mtp=[x for x in src.layers if x[1].startswith('mtp.')]; assert len(mtp)==1
    fields,trellis,rec=packer.layout(src)
    expected=[dict(name=n,offset=o,bytes=b,dtype=d,shape=list(s)) for n,o,b,d,s in fields]
    assert expected==meta['fields'] and rec==meta['record_bytes'] and trellis==meta['trellis_bytes']
    old_size=meta['total_bytes']; assert store.stat().st_size==old_size
    E=meta['experts_per_layer']; layer,prefix,count=mtp[0]; assert count==E
    # Check every MTP tensor's shape/codec against the trunk, before changing the store.
    for e in range(E):
        for name,_,n,dt,shape in fields:
            v=src.loc[f'{prefix}.experts.{e}.{name}']; assert v[2:]==(n,dt,shape),v
        for mat in packer.MATS:
            fn,off,n,dt,_=src.loc[f'{prefix}.experts.{e}.{mat}.mul1']
            assert os.pread(src.fds[fn],n,off).hex() in meta['mul1_markers_hex']
    fd=os.open(store,os.O_RDWR|os.O_DIRECT); os.posix_fallocate(fd,old_size,E*rec)
    local=threading.local(); t=time.monotonic()
    def job(e):
        if not hasattr(local,'buf'): local.buf=mmap.mmap(-1,rec)
        buf=local.buf; mv=packer.fill_record(src,prefix,e,fields,buf)
        end=fields[-1][1]+fields[-1][2]; mv[end:]=bytes(rec-end)
        digest=hashlib.sha256(mv).hexdigest(); assert os.pwrite(fd,mv,old_size+e*rec)==rec
        assert os.preadv(fd,[mv],old_size+e*rec)==rec
        assert hashlib.sha256(mv).hexdigest()==digest
        return digest
    with cf.ThreadPoolExecutor(12) as ex: hashes=list(ex.map(job,range(E)))
    os.fsync(fd); os.close(fd)
    li=meta['n_layers']; meta['layers'].append(dict(li=li,model_layer=layer,key=prefix))
    meta['mtp_layer_li']=li; meta['n_layers']+=1; meta['total_bytes']+=E*rec; meta['sha256']+=hashes
    tmp=manifest.with_suffix('.tmp'); tmp.write_text(json.dumps(meta)); tmp.replace(manifest)
    print('MTP_APPEND_VERIFIED',E,'experts',E*rec,'bytes',time.monotonic()-t,'seconds',flush=True)

if __name__=='__main__': main()
