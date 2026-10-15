import importlib.util
from pathlib import Path
import tempfile
import unittest


class ArtifactAuditTests(unittest.TestCase):
    def test_changed_policy_weights_are_rejected_even_when_launcher_is_unchanged(self):
        spec = importlib.util.spec_from_file_location('jpeg_audit', 'scripts/jpeg_strength_audit.py')
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory, 'engine')
            model = Path(directory, 'model.json')
            executable.write_bytes(b'unchanged executable')
            model.write_bytes(b'original policy')
            metadata = {
                'config': {'candidate': {'command': [str(executable)], 'artifacts': {'model': str(model)}}},
                'launcher_sha256': {'candidate': audit.sha(executable)},
                'declared_artifact_sha256': {'candidate': {'model': audit.sha(model)}},
            }
            audit.verify_engine_artifacts(metadata)
            model.write_bytes(b'different policy')
            with self.assertRaisesRegex(AssertionError, 'changed engine artifact'):
                audit.verify_engine_artifacts(metadata)


if __name__ == '__main__':
    unittest.main()
