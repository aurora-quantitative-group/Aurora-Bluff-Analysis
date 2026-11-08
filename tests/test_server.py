import http.client
import json
import threading
import unittest
from unittest.mock import patch

from server import Engine, Handler, ThreadingHTTPServer


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.engine = Engine()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port)
        connection.request(method, path, json.dumps(body) if body is not None else None, headers or {})
        response = connection.getresponse()
        status, data = response.status, response.read()
        connection.close()
        return status, data

    def test_assets_and_private_files(self):
        for path in ('/', '/app.js', '/style.css', '/assets/images/logo.webp', '/assets/fonts/coolvetica/Coolvetica%20Rg.otf'):
            self.assertEqual(self.request('GET', path)[0], 200)
        for path in ('/participants.json', '/deception_rf_model.pkl', '/../participants.json'):
            self.assertEqual(self.request('GET', path)[0], 404)

    def test_session_isolation(self):
        first = json.loads(self.request('POST', '/api/session', {})[1])['session_id']
        second = json.loads(self.request('POST', '/api/session', {})[1])['session_id']
        self.server.engine.get_session(first)['baseline'].append({'value': 1})
        self.assertEqual(self.server.engine.get_session(second)['baseline'], [])

    def test_invalid_input(self):
        self.assertEqual(self.request('POST', '/api/session', [1])[0], 400)
        self.assertEqual(self.request('POST', '/api/track', {'session_id': 'missing'})[0], 400)

    def test_cross_origin_blocked(self):
        self.assertEqual(self.request('POST', '/api/session', {}, {'Origin': 'https://example.com'})[0], 403)

    def test_tracking_requires_setup(self):
        engine = Engine()
        engine.binary = None
        key = engine.session()['session_id']
        with self.assertRaisesRegex(ValueError, 'OpenFace executable not found'):
            engine.track({'session_id': key, 'frames': []})

    def test_report_needs_key_and_observations(self):
        engine = Engine()
        key = engine.session()['session_id']
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'):
                engine.report({'session_id': key})
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test'}):
            with self.assertRaisesRegex(ValueError, 'Capture tracked'):
                engine.report({'session_id': key})

    def test_status_does_not_expose_key(self):
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'secret-value'}):
            status, body = self.request('GET', '/api/status')
            self.assertEqual(status, 200)
            self.assertNotIn(b'secret-value', body)
            self.assertTrue(json.loads(body)['gemini'])


if __name__ == '__main__':
    unittest.main()
