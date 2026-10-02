"""Cold-declared, CPU strong-teacher production for fresh licensed game families.

Import/admission/planning never searches or trains. Real label production requires
an independently sealed declaration. This milestone produces CPU score records;
GPU compact replay construction and learned comparisons are a later admission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import time

from .score_frames import score_frame_manifest
from .semantic_data import (ARCHIVE_SHA256, PGN_SHA256, GamePosition, StockfishSession,
    atomic_json, common_cp_labels, sha256_file)

KIND = 'chen_licensed_semantic_teacher_packet_v2_declaration'
STOCKFISH_SHA256 = '0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19'
from .ordinary_packets import POLICY

SOURCES = ('neural/ordinary_packets.py', 'neural/semantic_data.py', 'neural/semantic_study.py', 'neural/score_frames.py',
           'neural/semantic_strategy.py', 'neural/strategic_transport.py', 'neural/models.py',
           'neural/features.py', 'neural/compact.py', 'neural/bridge.py', 'neural/value_heads.py',
           'neural/dynamics.py', 'neural/diamonds.py', 'neural/teacher.py', 'neural/data.py', 'neural/train.py')
OPTIONS = dict(Threads=1, Hash=64, NumaPolicy='none', Ponder=False,
               UCI_LimitStrength=False, Skill_Level=20, UCI_ShowWDL=False,
               UCI_Chess960=False, SyzygyPath='', SyzygyProbeLimit=0)


class TeacherCancelled(RuntimeError):
    """Not EINTR: selectors/multiprocessing swallow InterruptedError retries."""


@dataclass(frozen=True)
class LabelBudget:
    standalone_nodes: int = 32000
    all_action_nodes: int = 256000
    workers: int = 2
    wall_seconds: int = 1200
    cleanup_seconds: int = 10
    request_seconds: int = 20
    min_reported_depth: int = 6

    def validate(self):
        if (any(type(v) is not int or v < 1 for v in asdict(self).values()) or
                self.workers > 2 or self.wall_seconds > 1200 or self.request_seconds > 20 or
                self.standalone_nodes > 32000 or self.all_action_nodes > 256000 or
                self.cleanup_seconds != 10 or self.wall_seconds <= self.cleanup_seconds):
            raise ValueError('teacher budget exceeds bounded CPU producer contract')


def read_admission(path):
    path = Path(path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    admission = json.loads(raw)
    if (admission.get('kind') != 'fresh_licensed_real_game_family_admission' or
            admission.get('teacher_labels_generated') is not False or admission.get('cross_split_overlap') != 0 or
            admission.get('spent_input_overlap') != 0 or admission.get('pgn_sha256') != PGN_SHA256 or
            admission.get('source', {}).get('archive_sha256') != ARCHIVE_SHA256):
        raise ValueError('unrecognized or exposed licensed admission')
    source = admission.get('source_records')
    if (not isinstance(source, dict) or set(source) != {'train', 'validation', 'test'} or
            hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest() != admission.get('source_sha256')):
        raise ValueError('admission record fingerprint differs')
    for split, families in source.items():
        if len(families) != admission['families'][split]:
            raise ValueError('family count differs from admitted split')
    return admission


def query_plan(admission, budget=LabelBudget()):
    """No search: stable IDs distinguish standalone and parent-normalized targets."""
    budget.validate()
    jobs, actions = [], 0
    import chess
    for split in ('train', 'validation', 'test'):
        for family in admission['source_records'][split]:
            for observed in family['positions']:
                payload = dict(game_id=observed['game_id'], ply=observed['ply'], fen=observed['fen'],
                               moves=observed['moves'], previous_hashes=observed['previous_hashes'])
                query = dict(split=split, position=payload, all_legal=False, nodes=budget.standalone_nodes)
                query['id'] = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
                jobs.append(query)
            observed = family['positions'][family['action_root_index']]
            payload = dict(game_id=observed['game_id'], ply=observed['ply'], fen=observed['fen'],
                           moves=observed['moves'], previous_hashes=observed['previous_hashes'])
            query = dict(split=split, position=payload, all_legal=True, nodes=budget.all_action_nodes)
            query['id'] = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
            jobs.append(query)
            actions += chess.Board(observed['fen']).legal_moves.count()
    return jobs, dict(queries=len(jobs), standalone_queries=sum(not j['all_legal'] for j in jobs),
                     all_action_queries=sum(j['all_legal'] for j in jobs), planned_action_rows=actions,
                     requested_nodes=sum(j['nodes'] for j in jobs),
                     requested_nodes_are_not_exact_actual_work=True)


def make_declaration(admission_path, binary, *, repo='.', budget=LabelBudget()):
    """Return DRAFT. The caller/parent seals exact bytes before any real search."""
    admission = read_admission(admission_path)
    jobs, workload = query_plan(admission, budget)
    root = Path(repo)
    if sha256_file(binary) != STOCKFISH_SHA256:
        raise ValueError('teacher binary differs from the pinned official SF19 universal build')
    for spent in admission['spent_reports']:
        if sha256_file(root/spent['path']) != spent['sha256']:
            raise ValueError('ordered spent-report bytes changed after admission')
    return dict(kind=KIND, version=2, status='draft_no_search_authorized', strength_evidence=False,
                admission_sha256=sha256_file(admission_path), admitted_source_sha256=admission['source_sha256'],
                query_plan_sha256=hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest(),
                spent_reports=admission['spent_reports'], families=admission['families'], workload=workload,
                source_sha256={name: sha256_file(root/name) for name in SOURCES},
                stockfish=dict(binary_sha256=sha256_file(binary), expected_name_prefix='Stockfish 19',
                               tag='sf_19', options=OPTIONS, clear_hash_each_query='ucinewgame_then_isready',
                               release_url='https://github.com/official-stockfish/Stockfish/releases/tag/sf_19',
                               source_license='GPL-3.0-or-later; root match-tools preserves source/COPYING',
                               history='position_startpos_moves_full_game_prefix', labels='selective_finite_node_UCI_PV_estimates'),
                limits=asdict(budget), score_frame=score_frame_manifest(),
                retirement=dict(work_seconds=budget.wall_seconds-budget.cleanup_seconds,
                                reserved_cleanup_seconds=budget.cleanup_seconds, graceful_seconds=3,
                                ceiling_semantics='cooperative_work_retirement_checks_not_preemptive_CPU_IO_guarantee',
                                scope='Linux_exclusive_synchronous_producer_owned_POSIX_groups',
                                hard_escalation='SIGKILL_owned_groups_then_subreaper_waitpid_negative_pgid',
                                parent_signals='scoped_SIGTERM_SIGINT_unwind_reentrant_ignored_during_retirement',
                                subreaper='restore_original_process_setting_after_all_owned_children_reaped'),
                protocol=dict(packet_policy=POLICY, common_packet=POLICY['selection'],
                              rejected='any_decisive_evidence_or_protocol_defect; no_eligible_ordinary_packet',
                              root_coverage='whole_all_legal_root_or_reject_no_replacement',
                              family_coverage='all_eight_standalone_queries_and_complete_root_or_reject',
                              label_query_kinds='standalone_root_vs_parent_MultiPV_distinct_even_after_common_frame_conversion',
                              context_caveat='converted_units_comparable; search_context_budget_and_reported_depth_differ',
                              outcome_labels=False, ordinary_min_abs_cp=None,
                              cpu_labels_ingress='explicit_CPU_UCI_records_not_GPU_teacher_payload'),
                model_admission=dict(module='neural.semantic_strategy', encoders=['square','relation','plan','threat'],
                                     feature_schema=1, width=32, depth=1,
                                     representation='train_only_semantic_decoder_and_value_pretraining_then_frozen_shared_readout',
                                     anchor_calibration='common_positive_material_scale_fit_train_standalone_only',
                                     original_material_baseline='pawn100_no_fit_retained',
                                     successor_ingress='native_GPU_bridge_exact_records_after_separate_GPU_admission',
                                     blind_cache='one_root_anchor_and_residual_per_sibling_group',
                                     training_authorized=False))


def validate_declaration(declaration_path, admission_path, binary, *, repo='.'):
    raw = Path(declaration_path).read_bytes(); declaration = json.loads(raw)
    if declaration.get('status') != 'sealed_for_bounded_labels':
        raise ValueError('real search requires independently sealed bounded-label declaration')
    limits = declaration.get('limits')
    if not isinstance(limits, dict) or set(limits) != set(asdict(LabelBudget())):
        raise ValueError('teacher declaration limit keys differ')
    budget = LabelBudget(**limits); budget.validate()
    expected = make_declaration(admission_path, binary, repo=repo, budget=budget)
    expected['status'] = 'sealed_for_bounded_labels'
    # JSON is type-sensitive; bool aliases cannot impersonate numeric budgets.
    if json.dumps(declaration, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('cold source/data/game-history/options/binary/workload declaration differs')
    admission = read_admission(admission_path)
    return declaration, hashlib.sha256(raw).hexdigest(), admission


def _teacher_worker(index, jobs, out, binary, binary_sha, declaration_sha, budget, deadline):
    """Dedicated POSIX process group owns its single Stockfish child."""
    os.setsid()
    def terminate(signum, frame):
        # Re-entering a handler during close/wait could abandon a grandchild.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        raise TeacherCancelled('teacher worker retirement requested')
    signal.signal(signal.SIGTERM, terminate)
    # Spawn inherits the parent's brief ownership-registration signal mask.
    # Install cleanup handlers FIRST, then allow cancellation before any engine.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    current, session = None, None
    try:
        with StockfishSession(binary, expected_sha256=binary_sha, timeout=budget.request_seconds) as session:
            atomic_json(Path(out)/f'worker-{index}-setup.json', dict(identity=session.identity, trace=session.setup_trace,
                        worker_pid=os.getpid(), engine_pid=session.process.pid,
                        blocked_cancellation_signals=[sig.name for sig in
                            signal.pthread_sigmask(signal.SIG_BLOCK, set()) & {signal.SIGTERM, signal.SIGINT}]))
            for job in jobs:
                current = job['id']
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    break
                session.timeout = min(budget.request_seconds, remaining)
                observed = job['position']
                pos = GamePosition(observed['game_id'], observed['ply'], observed['fen'],
                                   tuple(observed['moves']), tuple(observed['previous_hashes']))
                start = time.monotonic()
                lines = session.analyze(pos, nodes=job['nodes'], all_legal=job['all_legal'], declaration_sha256=declaration_sha)
                packet_provenance = {}
                labels, reason = common_cp_labels(lines, pos, all_legal=job['all_legal'], min_depth=budget.min_reported_depth, diagnostics=packet_provenance)
                record = dict(query=job, labels=labels, rejection=reason, packet_provenance=packet_provenance, transcript=session.take_trace(),
                              elapsed_seconds=time.monotonic()-start, declaration_sha256=declaration_sha)
                atomic_json(Path(out)/(job['id']+'.json'), record)
        if session.process.returncode != 0:
            raise RuntimeError('teacher engine failed during final close')
    except BaseException as error:
        atomic_json(Path(out)/f'worker-{index}-error.json', dict(type=type(error).__name__, message=str(error),
                    query_id=current, partial_transcript=[] if session is None else session.take_trace()))
        raise


class _OwnedSubreaper:
    """Temporary Linux admission; reap ONLY the producer's dedicated groups.

    This synchronous producer requires exclusive subprocess lifecycle access in
    its calling process. The prior subreaper flag is restored, and unrelated
    process groups are never waited/killed. Fail closed if prctl is unavailable.
    """
    def __init__(self):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.original = ctypes.c_int()
        if self.libc.prctl(37, ctypes.byref(self.original), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Linux child-subreaper query failed')
        if self.libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Linux child-subreaper admission failed')

    def reap_group(self, pgid, deadline):
        reaped = []
        while True:
            try:
                pid, status = os.waitpid(-pgid, os.WNOHANG)
            except ChildProcessError:
                return reaped
            if pid:
                reaped.append(dict(pid=pid, wait_status=status))
                continue
            if time.monotonic() >= deadline:
                raise RuntimeError('owned teacher child remained unreaped at retirement ceiling')
            time.sleep(.01)

    def close(self):
        if self.libc.prctl(36, self.original.value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Linux child-subreaper restoration failed')


class _ParentRetirementSignals:
    """Exclusive main-thread producer; preserve caller's handlers on every exit."""
    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise ValueError('teacher producer requires exclusive main-thread signal lifecycle')
        if signal.pthread_sigmask(signal.SIG_BLOCK, set()) & {signal.SIGTERM, signal.SIGINT}:
            raise ValueError('teacher producer requires unblocked caller cancellation signals')
        self.original = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in self.original: signal.signal(sig, self.cancel)
        return self

    def cancel(self, signum, frame):
        self.retire()
        raise TeacherCancelled('bounded teacher producer cancelled by signal ' + str(signum))

    def retire(self):
        for sig in self.original: signal.signal(sig, signal.SIG_IGN)

    def __exit__(self, *error):
        for sig, handler in self.original.items(): signal.signal(sig, handler)


def produce_labels(declaration_path, admission_path, binary, output, *, repo='.'):
    with _ParentRetirementSignals() as signals:
        return _produce_labels(declaration_path, admission_path, binary, output, repo=repo, signals=signals)


def _produce_labels(declaration_path, admission_path, binary, output, *, repo, signals):
    """Two CPU workers, immutable query set, hard whole-job stop; no GPU or training.

    Partial output remains a closed attempt. A timeout never expands the query
    set or continues into another budget. Every child process is reaped.
    """
    started = time.monotonic()
    declaration, declaration_sha, admission = validate_declaration(declaration_path, admission_path, binary, repo=repo)
    budget = LabelBudget(**declaration['limits']); jobs, workload = query_plan(admission, budget)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('new label attempt requires an empty output directory')
    atomic_json(out/'declaration.json', declaration)
    retirement_deadline = started+budget.wall_seconds
    deadline = retirement_deadline-budget.cleanup_seconds
    atomic_json(out/'report.json', dict(kind='licensed_finite_search_teacher_attempt', status='running',
                declaration_sha256=declaration_sha, strength_evidence=False, requested_queries=len(jobs)))
    context = mp.get_context('spawn')
    workers = [context.Process(target=_teacher_worker, args=(i, jobs[i::budget.workers], str(out), str(Path(binary).resolve()),
                declaration['stockfish']['binary_sha256'], declaration_sha, budget, deadline)) for i in range(budget.workers)]
    started_workers = []
    timed_out, interruption = False, None
    subreaper = None
    adopted = []
    try:
        subreaper = _OwnedSubreaper()
        for process in workers:
            # A signal must not land between successful spawn and ownership
            # registration, otherwise the new child could escape retirement.
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
            try:
                process.start(); started_workers.append(process)
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        while any(process.is_alive() for process in started_workers):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                timed_out = True; break
            for process in started_workers:
                process.join(timeout=min(.1, remaining))
    except BaseException as error:
        interruption = error
    finally:
        signals.retire()
        try:
            # First ask workers only: their SIGTERM handler runs close/wait.
            for process in started_workers:
                if process.is_alive(): process.terminate()
            graceful_deadline = min(retirement_deadline, time.monotonic()+3)
            while any(p.is_alive() for p in started_workers) and time.monotonic() < graceful_deadline:
                for process in started_workers:
                    process.join(timeout=max(0, min(.05, graceful_deadline-time.monotonic())))
            # Always kill any remaining owned group, including an orphan from
            # an already-exited worker; subreaper makes it our waitable child.
            for process in started_workers:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    if process.is_alive(): process.kill()
            for process in started_workers:
                process.join(timeout=max(0, retirement_deadline-time.monotonic()))
                if process.is_alive(): raise RuntimeError('teacher worker could not be reaped')
                adopted.extend(subreaper.reap_group(process.pid, retirement_deadline))
        finally:
            if subreaper is not None: subreaper.close()
    records, family_queries, finalization_timed_out, fingerprints = [], {}, False, []
    for job in jobs:
        key = (job['split'], job['position']['game_id'])
        file = out/(job['id']+'.json')
        if time.monotonic() >= retirement_deadline:
            finalization_timed_out = True
            record = None
        else:
            record = json.loads(file.read_text()) if file.exists() else None
        family_queries.setdefault(key, []).append(record)
        if record is not None:
            records.append(record)
            fingerprints.append(dict(query_id=job['id'], sha256=sha256_file(file)))
    mechanically_complete = {split: [] for split in ('train','validation','test')}
    accepted = {split: [] for split in mechanically_complete}
    rejected = {split: [] for split in accepted}
    for (split, game_id), rows in family_queries.items():
        if all(row is not None and row['labels'] is not None for row in rows):
            mechanically_complete[split].append(game_id)
    if time.monotonic() >= retirement_deadline:
        finalization_timed_out = True
    attempt_complete = (interruption is None and not timed_out and not finalization_timed_out and
                        len(records) == len(jobs) and len(started_workers) == len(workers) and
                        all(p.exitcode == 0 for p in started_workers))
    for split, game_id in family_queries:
        if attempt_complete and game_id in mechanically_complete[split]:
            accepted[split].append(game_id)
        else:
            rejected[split].append(game_id)
    result = dict(kind='licensed_finite_search_teacher_attempt', strength_evidence=False,
                  declaration_sha256=declaration_sha, admission_sha256=sha256_file(admission_path),
                  status='partial_closed_interruption' if interruption is not None else
                         ('partial_closed_finalization_ceiling' if finalization_timed_out else
                         ('partial_closed_wall_ceiling' if timed_out else
                         ('complete' if attempt_complete else 'partial_closed_worker_failure'))),
                  completed_queries=len(records), requested_queries=len(jobs), workload=workload,
                  wall_seconds=time.monotonic()-started, worker_exitcodes=[p.exitcode for p in started_workers],
                  work_budget_seconds=budget.wall_seconds-budget.cleanup_seconds,
                  reserved_cleanup_seconds=budget.cleanup_seconds, adopted_children_reaped=adopted,
                  whole_family_accepted=accepted, whole_family_rejected=rejected,
                  mechanically_complete_families_not_admitted_if_attempt_failed=mechanically_complete,
                  partial_attempt_admission='none_all_outputs_quarantined_no_automatic_retry',
                  source_records=fingerprints,
                  next_stage='separate_source_and_label_admission_before_GPU_corpus_or_training')
    atomic_json(out/'report.json', result)
    if interruption is not None:
        raise interruption
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'labels'))
    parser.add_argument('--admission', required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--declaration')
    args = parser.parse_args()
    if args.stage == 'plan':
        if args.declaration is not None:
            parser.error('plan produces a draft; it does not load a sealed declaration')
        result = make_declaration(args.admission, args.binary, repo=args.repo)
        atomic_json(args.output, result)
        print(json.dumps(dict(status=result['status'], workload=result['workload']), indent=2))
    else:
        if args.declaration is None:
            parser.error('labels requires an independently sealed declaration')
        result = produce_labels(args.declaration, args.admission, args.binary, args.output, repo=args.repo)
        print(json.dumps({k:v for k,v in result.items() if k != 'source_records'}, indent=2))


if __name__ == '__main__':
    main()
