"""Check actual SSE answer streaming with tools present, as in Pi."""
import argparse,json,time,urllib.request
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--label',required=True);args=ap.parse_args()
body={'model':'qwen3.8-flash-next','messages':[{'role':'user','content':
      'Write one paragraph of about 90 words explaining how bicycles work. Do not call any tools.'}],
      'tools':[{'type':'function','function':{'name':'multiply','description':'Multiply two integers.',
      'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}],
      'temperature':0.0,'max_tokens':512,'stream':True,
      'chat_template_kwargs':{'enable_thinking':True,'reasoning_effort':'low'}}
start=time.monotonic();chunks=[];done=False
req=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body).encode(),
                           headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=180) as response:
    for line in response:
        if not line.startswith(b'data: '):continue
        raw=line[6:].strip()
        if raw==b'[DONE]':done=True;break
        event=json.loads(raw)
        chunks.append({'seconds':time.monotonic()-start,'event':event})
content=[r for r in chunks if r['event']['choices'][0].get('delta',{}).get('content')]
reasoning=[r for r in chunks if r['event']['choices'][0].get('delta',{}).get('reasoning_content')]
result={'done':done,'total_seconds':time.monotonic()-start,'reasoning_chunks':len(reasoning),
        'content_chunks':len(content),'first_content_seconds':content[0]['seconds'] if content else None,
        'last_content_seconds':content[-1]['seconds'] if content else None,
        'answer':''.join(r['event']['choices'][0]['delta']['content'] for r in content),'chunks':chunks}
Path(f'research/local-server/answer-stream-{args.label}.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='chunks'}),flush=True)
assert done and reasoning and content,result
if args.label=='after':assert len(content)>1 and content[-1]['seconds']>content[0]['seconds'],result
