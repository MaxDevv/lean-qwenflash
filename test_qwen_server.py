"""Real HTTP streaming and executed-tool round-trip checks against the local model."""
import json,time,urllib.request
from pathlib import Path
root=Path('research/local-server');out=root/'api-validation.json';results={}
def save():out.write_text(json.dumps(results,indent=2))
def get(path):
 with urllib.request.urlopen('http://127.0.0.1:30000'+path,timeout=120) as r:return json.load(r)
def post(body):
 req=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)
results['models']=get('/v1/models');results['server_info']=get('/server_info');save()
model='qwen3.8-flash-next'
tools=[{'type':'function','function':{'name':'multiply','description':'Multiply two integers and return their product.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}]
messages=[{'role':'user','content':'Use the multiply tool to calculate 17 times 23. You must call the tool first; after receiving its result, reply with only the product.'}]
body={'model':model,'messages':messages,'tools':tools,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
results['tool_request']=post(body);save();message=results['tool_request']['choices'][0]['message']
calls=message.get('tool_calls',[]);assert len(calls)==1,results['tool_request']
call=calls[0];args=json.loads(call['function']['arguments']);assert call['function']['name']=='multiply' and args=={'a':17,'b':23},call
product=args['a']*args['b']
messages.extend([message,{'role':'tool','tool_call_id':call['id'],'content':json.dumps({'product':product})}])
results['tool_response']=post(body);save()
answer=results['tool_response']['choices'][0]['message'];assert str(product) in answer['content'] and not answer.get('tool_calls'),results['tool_response']
stream_body={'model':model,'messages':[{'role':'user','content':'Reply with exactly STREAM_OK.'}],'temperature':0,'stream':True,'chat_template_kwargs':{'enable_thinking':False}}
req=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(stream_body).encode(),headers={'Content-Type':'application/json'})
chunks=[];done=False;content=''
with urllib.request.urlopen(req,timeout=120) as r:
 for line in r:
  line=line.decode().strip()
  if not line.startswith('data: '):continue
  data=line[6:]
  if data=='[DONE]':done=True;break
  event=json.loads(data);chunks.append(event)
  content+=event['choices'][0].get('delta',{}).get('content','') or ''
results['stream']={'content':content,'done':done,'chunks':chunks};results['memory']=get('/memory');save()
assert done and 'STREAM_OK' in content,results['stream']
results['passed']=True;save();print(json.dumps({'tool_product':product,'tool_final':answer['content'],'stream':content,'memory':results['memory']}),flush=True)
