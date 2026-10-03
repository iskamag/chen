"""Exact failed-parent boundary admission, CPU-only; no numerical re-execution."""
import copy
from pathlib import Path
import tempfile
import unittest

from neural.profile_precision_capture import PARENT_PINS, admit_continuation


class ContinuationFixtures(unittest.TestCase):
    def fixture(self):
        archive = Path(__file__).resolve().parents[1] / 'evidence/precision-capture-partial-gfx1030-2026-10-03'
        files = dict(declaration=archive / 'control/declaration.json', terminal=archive / 'control/watchdog-terminal.json',
                     square_partial=archive / 'attempt-1/anchor-square-20261021.json')
        keys = [f'{e}/20261021' for e in ('relation', 'plan', 'threat')]
        keys += [f'{e}/conditioned/20261021' for e in ('square', 'relation', 'plan', 'threat')]
        return dict(kind='chen_train_precision_capture_continuation_v1', checkpoint_keys=keys,
                    parent_attempt={name:dict(path=str(path), sha256=PARENT_PINS[name]) for name, path in files.items()})

    def test_actual_closed_parent_admits_exact_remaining_keys(self):
        fixture = self.fixture()
        self.assertEqual(admit_continuation(fixture), fixture['checkpoint_keys'])
        self.assertNotIn('square/20261021', fixture['checkpoint_keys'])

    def test_no_generic_skip_replay_or_parent_substitution(self):
        fixture = self.fixture()
        for keys in (fixture['checkpoint_keys'][:-1], ['square/20261021'] + fixture['checkpoint_keys'],
                     list(reversed(fixture['checkpoint_keys']))):
            changed = copy.deepcopy(fixture)
            changed['checkpoint_keys'] = keys
            with self.assertRaisesRegex(ValueError, 'seven unfinished'):
                admit_continuation(changed)
        changed = copy.deepcopy(fixture)
        changed['parent_attempt']['terminal']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'parent witness'):
            admit_continuation(changed)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tampered-terminal.json'
            path.write_text('{}')
            changed = copy.deepcopy(fixture)
            changed['parent_attempt']['terminal']['path'] = str(path)
            with self.assertRaisesRegex(ValueError, 'parent witness'):
                admit_continuation(changed)


if __name__ == '__main__':
    unittest.main()
