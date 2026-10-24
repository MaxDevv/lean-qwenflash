import importlib.util,json,struct,tempfile,unittest
from pathlib import Path

path=Path(__file__).resolve().parents[1]/'glm53-flash-offload/scripts/pack_glm53_store.py'
spec=importlib.util.spec_from_file_location('packer',path); packer=importlib.util.module_from_spec(spec); spec.loader.exec_module(packer)

class StoreGeometry(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        header={}; offset=0
        for prefix in ('model.language_model.layers.0.mlp','mtp.layers.0.mlp'):
            for ex in range(2):
                for mat in packer.MATS:
                    hin,hout=(640,2560) if mat=='down_proj' else (2560,640)
                    fields={'trellis':('I16',[hin//16,hout//16,48],hin*hout*3//8),
                            'suh':('F16',[hin],hin*2),'svh':('F16',[hout],hout*2),
                            'mul1':('I32',[],4)}
                    for suffix,(dtype,shape,size) in fields.items():
                        key=f'{prefix}.experts.{ex}.{mat}.{suffix}'
                        header[key]={'dtype':dtype,'shape':shape,'data_offsets':[offset,offset+size]}; offset+=size
        payload=json.dumps(header).encode(); fn='model.safetensors'
        with (self.root/fn).open('wb') as f:
            f.write(struct.pack('<Q',len(payload))); f.write(payload); f.truncate(8+len(payload)+offset)
        (self.root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{k:fn for k in header}}))
    def tearDown(self): self.temp.cleanup()
    def test_mtp_does_not_collide_with_trunk_layer_zero(self):
        src=packer.Src(str(self.root),True)
        self.assertEqual([x[1] for x in src.layers],['model.language_model.layers.0.mlp','mtp.layers.0.mlp'])
    def test_text_only_excludes_mtp_and_preserves_qwen_geometry(self):
        src=packer.Src(str(self.root),False)
        self.assertEqual(len(src.layers),1)
        fields,trellis,record=packer.layout(src)
        self.assertEqual(trellis,1843200); self.assertEqual(record,1863680)
        self.assertEqual([f[1] for f in fields[:3]],[0,614400,1228800])
        self.assertEqual(packer.check_uniform(src,fields,1),['00000000'])

if __name__=='__main__': unittest.main()
