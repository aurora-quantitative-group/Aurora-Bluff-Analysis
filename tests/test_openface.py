import base64
import os
import unittest
from pathlib import Path

from server import Engine


@unittest.skipUnless(os.environ.get('AURORA_TEST_OPENFACE') == '1', 'Set AURORA_TEST_OPENFACE=1 to run native tracking')
class OpenFaceTests(unittest.TestCase):
    def test_calibration_and_scoring(self):
        engine = Engine()
        self.assertTrue(engine.status()['openface'])
        self.assertTrue(engine.status()['classifier'])
        directory = Path(__file__).resolve().parents[1] / '.local/OpenFace/samples/image_sequence'
        paths = sorted(directory.glob('*.jpg'))[:20]
        self.assertEqual(len(paths), 20)
        frames = [base64.b64encode(path.read_bytes()).decode() for path in paths]
        key = engine.session()['session_id']
        request = {'session_id': key, 'frames': frames, 'fps': 10, 'calibrating': True}
        for count in range(1, 4):
            result = engine.track(request)
            self.assertTrue(result['tracked'], result)
            self.assertEqual(result['baseline_count'], count)
            self.assertIsNone(result['score'])
            self.assertGreater(result['box']['width'], 0)
        result = engine.track({**request, 'calibrating': False})
        self.assertTrue(result['tracked'], result)
        self.assertGreaterEqual(result['score'], 0)
        self.assertLessEqual(result['score'], 1)
        self.assertEqual(len(engine.get_session(key)['samples']), 4)
        print(f"OpenFace tracked {len(frames)} frames per batch; classifier score {result['score']:.3f}")


if __name__ == '__main__':
    unittest.main()
