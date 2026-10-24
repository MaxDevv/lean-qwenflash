"""Matched HTTP prefix-reuse regression, including a growing conversation."""
import argparse,json,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);args=ap.parse_args()
BASE='http://127.0.0.1:30000'
def request(path,body=None):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
for _ in range(120):
    try:request('/health');break
    except OSError:time.sleep(1)
else:raise RuntimeError('Server not ready')
source=(ROOT/'research/moby-dick.txt')
if not source.exists():
    source=ROOT/'model/README.md'
text=source.read_text()
# Tokenize on the actual serving tokenizer, grow/truncate text to approximately 10k tokens.
text=(text*8)[:60000]
lo,hi=0,len(text)
while lo<hi:
    mid=(lo+hi+1)//2
    if len(request('/tokenize',{'prompt':text[:mid]})['tokens'])<=10000:lo=mid
    else:hi=mid-1
messages=[{'role':'system','content':'Answer very briefly.'},
          {'role':'user','content':text[:lo]+'\nReply with only the digit 4.'}]
results=[]
path=ROOT/'research/local-server'/f'prefix-cache-{args.label}.json'
for label in ('first','identical','followup','followup-identical'):
    if label=='followup':
        messages=messages+[{'role':'assistant','content':results[-1]['answer']},
                           {'role':'user','content':'Now reply with only the digit 5.'}]
    payload={'model':'qwen3.8-flash-next','messages':messages,'max_tokens':8,
             'temperature':0.0,'chat_template_kwargs':{'enable_thinking':False},'stream':True}
    start=time.monotonic();first=None;answer='';reason=None
    req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            raw=line[6:].strip()
            if raw==b'[DONE]':break
            event=json.loads(raw);choice=event['choices'][0];delta=choice.get('delta',{})
            if delta.get('content'):
                if first is None:first=time.monotonic()-start
                answer+=delta['content']
            if choice.get('finish_reason'):reason=choice['finish_reason']
    row={'request':label,'ttft_seconds':first,'total_seconds':time.monotonic()-start,
         'answer':answer,'finish_reason':reason,'server':request('/stats').get('last_request_prefill')}
    results.append(row);path.write_text(json.dumps(results,indent=2));print(label,json.dumps(row),flush=True)
    assert first is not None,row
    assert answer.strip()==('5' if label.startswith('followup') else '4'),row
    if args.label=='after':
        assert row['server']['recurrent_checkpoints']>0,row
        if label!='first':assert row['server']['cached_tokens']>=9984,row
