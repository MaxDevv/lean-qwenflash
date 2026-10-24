"""Executed-tool round trip through the changed SSE path, not non-streaming HTTP."""
import json,urllib.request
from pathlib import Path
out=Path('research/local-server/stream-tool-validation.json');results=[]
tools=[{'type':'function','function':{'name':'multiply','description':'Multiply two integers.',
        'parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b']}}}]
messages=[{'role':'user','content':'Use the multiply tool to calculate 17 times 23. You must call the tool first; after receiving its result, reply with only the product.'}]
def stream():
    body={'model':'qwen3.8-flash-next','messages':messages,'tools':tools,'stream':True,'temperature':0.0,
          'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request('http://127.0.0.1:30000/v1/chat/completions',data=json.dumps(body).encode(),
                               headers={'Content-Type':'application/json'})
    chunks=[];content='';calls=[];done=False;finish=None
    with urllib.request.urlopen(req,timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            raw=line[6:].strip()
            if raw==b'[DONE]':done=True;break
            event=json.loads(raw);chunks.append(event)
            choice=event['choices'][0];delta=choice.get('delta',{})
            content+=delta.get('content') or ''
            calls.extend(delta.get('tool_calls') or [])
            finish=choice.get('finish_reason') or finish
    result={'content':content,'tool_calls':calls,'done':done,'finish_reason':finish,'chunks':chunks}
    results.append(result);out.write_text(json.dumps(results,indent=2))
    assert done and '<tool_call>' not in content,result
    return result
first=stream();assert first['finish_reason']=='tool_calls' and len(first['tool_calls'])==1,first
call=first['tool_calls'][0];call.pop('index',None)
arguments=json.loads(call['function']['arguments'])
assert call['function']['name']=='multiply' and arguments=={'a':17,'b':23},call
product=arguments['a']*arguments['b']
messages.extend([{'role':'assistant','content':first['content'],'tool_calls':[call]},
                 {'role':'tool','tool_call_id':call['id'],'content':json.dumps({'product':product})}])
second=stream();assert second['content'].strip()=='391' and not second['tool_calls'],second
print('SSE tool call + executed result + final answer passed:',second['content'],flush=True)
