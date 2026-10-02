"""Cold metadata-only admission check; no heldout quality metrics or training."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from neural import run_semantic_corpus_study as r
p=r.read_json('results/semantic-corpus-study/outer-runner/host-unsealed-proposal.json')
args=(p['main_root'],Path(p['splits']['train']['path']).parent,p['kernel_declaration']['path'],
      p['baseline_witness']['path'],p['baseline_witness']['sha256'],p['attempt_output'])
proposal=r.suite_proposal(*args)
r.validate_witness(r.read_json(p['baseline_witness']['path'],p['baseline_witness']['sha256']),proposal['splits'])
with TemporaryDirectory()as td:
    kernel=r.read_json(args[2])
    for split in ('train','validation'):
        bad=kernel.copy();bad[split+'_manifest_sha256']='0'*64
        changed=Path(td)/('kernel-'+split+'.json'); changed.write_text(json.dumps(bad))
        try:r.suite_proposal(args[0],args[1],changed,*args[3:])
        except ValueError as error:
            assert 'corpus identities differ' in str(error)
        else:raise AssertionError('mismatched corpus admitted')
        print(json.dumps({'probe':'actual_kernel_'+split+'_mismatch','rejected':True}))
proposal_path=Path('results/semantic-corpus-study/outer-runner/host-reviewed-unsealed-proposal.json')
print(json.dumps({'proposal_path':str(proposal_path),'proposal_sha256':r.publish_json(proposal_path,proposal),
                  'baseline_structural_admission':True,'heldout_quality_metrics':False,'training':False}))
