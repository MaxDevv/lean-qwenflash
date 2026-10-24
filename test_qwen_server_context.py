"""Near-full-context retrieval through the actual OpenAI streaming endpoint."""
import json,time,urllib.request
from pathlib import Path
from tokenizers import Tokenizer
root=Path('research/local-server/validation');root.mkdir(exist_ok=True)
out=root/'context.json';result={'expected':['739164','285903','614827']}
def save():out.write_text(json.dumps(result,indent=2))
tok=Tokenizer.from_file('model/tokenizer.json')
source=Path('research/local-verification/moby-dick.txt').read_text()
ids=tok.encode(source).ids
if len(ids)<261120:ids=(ids*(1+261120//len(ids)))
ids=ids[:261120]
for fraction,name,code in ((0.1,'ARCTIC','739164'),(0.5,'DESERT','285903'),(0.9,'FOREST','614827')):
 entry=tok.encode(f'\nThe secret access code for {name} is {code}.\n').ids
 position=int(len(ids)*fraction);ids[position:position+len(entry)]=entry
content='Read this document carefully and retain its access codes.\n'+tok.decode(ids,skip_special_tokens=False)+'\nWhat are the secret access codes for ARCTIC, DESERT, and FOREST? Reply only with the three codes in that order.'
body={'model':'qwen3.8-flash-next','messages':[{'role':'user','content':content}],'temperature':0,'stream':True,'chat_template_kwargs':{'enable_thinking':False}}
result['filler_tokens']=len(ids);result['started']=time.time();save()
request=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
t0=time.perf_counter();text='';first=None;done=False
with urllib.request.urlopen(request,timeout=1800) as response:
 for line in response:
  line=line.decode().strip()
  if not line.startswith('data: '):continue
  payload=line[6:]
  if payload=='[DONE]':done=True;break
  event=json.loads(payload);choice=event['choices'][0];new=choice.get('delta',{}).get('content','') or ''
  if new and first is None:first=time.perf_counter()
  text+=new
  if event.get('usage'):result['usage']=event['usage']
  if choice.get('finish_reason'):result['finish_reason']=choice['finish_reason']
  if new or event.get('usage'):
   result['output']=text;result['elapsed_seconds']=time.perf_counter()-t0;save()
result.update(output=text,done=done,wall_seconds=time.perf_counter()-t0,first_content_seconds=first-t0 if first else None,
              passed=done and all(code in text for code in result['expected']))
save();assert result['passed'],result
assert result['usage']['prompt_tokens']>=260000,result['usage']
print(json.dumps(result),flush=True)
print('LOCAL_VERIFICATION_DONE',flush=True)
