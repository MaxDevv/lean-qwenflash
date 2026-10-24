"""Pure regression tests for prose/tool-call separation at arbitrary chunk boundaries."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('tool_stream',Path(__file__).parent/'qwen38-flash-offload/qwen38/tool_stream.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
split=module.split_tool_stream

class ToolStreamTests(unittest.TestCase):
    def feed(self,chunks):
        emitted='';held=''
        for chunk in chunks:
            safe,held=split(held+chunk)
            emitted+=safe
            self.assertNotIn('<tool_call>',emitted)
        return emitted,held
    def test_plain_prose_streams(self):
        text='Hello world. Unicode: café 日本語. Math: x < 2.\n'
        self.assertEqual(self.feed(list(text)),(text,''))
    def test_all_two_chunk_splits(self):
        prefix='Let me check.\n'
        call='<tool_call>\n<function=multiply><parameter=a>17</parameter><parameter=b>23</parameter></function>\n</tool_call>'
        for tail in ('','\nDone.','\n'+call):
            text=prefix+call+tail
            for at in range(len(text)+1):
                self.assertEqual(self.feed([text[:at],text[at:]]),(prefix,call+tail))
            self.assertEqual(self.feed(list(text)),(prefix,call+tail))
    def test_partial_marker_is_retained(self):
        for size in range(1,len('<tool_call>')):
            partial='<tool_call>'[:size]
            self.assertEqual(split('Answer '+partial),('Answer ',partial))
    def test_false_prefix_released(self):
        self.assertEqual(self.feed(['Hello <tool','box> world']),('Hello <toolbox> world',''))

if __name__=='__main__':unittest.main()
