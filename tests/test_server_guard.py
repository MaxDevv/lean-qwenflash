"""qwen_server's request guard: no web pages, and on localhost no other host names (DNS rebinding)."""
import asyncio
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LocalOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # importing qwen_server loads no engine; only jinja2 is stubbed when it isn't installed
        sandbox = types.ModuleType("jinja2.sandbox")
        sandbox.ImmutableSandboxedEnvironment = object
        try:
            import jinja2.sandbox  # noqa: F401
            stubs = {}
        except ImportError:
            stubs = {"jinja2": types.ModuleType("jinja2"), "jinja2.sandbox": sandbox}
        with mock.patch.dict(sys.modules, stubs):
            import qwen_server
        cls.LocalOnly = qwen_server.LocalOnly

    def call(self, headers, local=True, origins=(), hosts=()):
        """'app' when the request reaches the server, else the refusal's status."""
        reached, sent = [], []

        async def app(scope, receive, send):
            reached.append(scope)

        async def send(message):
            sent.append(message)
        scope = {"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
        asyncio.run(self.LocalOnly(app, local, origins, hosts)(scope, None, send))
        return "app" if reached else sent[0]["status"]

    def test_allowed_origins_and_hosts(self):
        ext = {"host": "127.0.0.1:30000", "origin": "chrome-extension://abc"}
        self.assertEqual(self.call(ext, origins=["chrome-extension://abc"]), "app")
        self.assertEqual(self.call(ext, origins=["*"]), "app")
        self.assertEqual(self.call(ext, origins=["https://other.example"]), 403)
        proxied = {"host": "ai.example.ts.net"}
        self.assertEqual(self.call(proxied), 403)
        self.assertEqual(self.call(proxied, hosts=["AI.example.ts.net"]), "app")

    def test_bound_to_localhost(self):
        self.assertEqual(self.call({"host": "127.0.0.1:30000"}), "app")
        self.assertEqual(self.call({"host": "localhost:30000"}), "app")
        self.assertEqual(self.call({"host": "[::1]:30000"}), "app")
        self.assertEqual(self.call({"host": "rebind.example:30000"}), 403)
        self.assertEqual(self.call({"host": "127.0.0.1:30000", "origin": "https://site.example"}), 403)

    def test_bound_to_a_network_address(self):
        self.assertEqual(self.call({"host": "192.168.1.20:30000"}, local=False), "app")
        self.assertEqual(self.call({"host": "192.168.1.20:30000", "origin": "https://site.example"}, local=False), 403)

    def test_lifespan_events_pass(self):
        reached = []

        async def app(scope, receive, send):
            reached.append(scope["type"])
        asyncio.run(self.LocalOnly(app)({"type": "lifespan"}, None, None))
        self.assertEqual(reached, ["lifespan"])


if __name__ == "__main__":
    unittest.main()
