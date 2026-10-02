"""Prepare metadata/source declarations only; never train or evaluate labels."""
import hashlib,json,sys
from pathlib import Path
base=Path(sys.argv[1]).resolve()
sys.path[:0]=[str(base/'neural-worktree'),str(base/'main'),str(base)]
from neural import semantic_corpus_study as study
from neural import run_semantic_corpus_study as runner
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
transport=runner.read_json(base/'control/host-transport.json')
for row in transport['sources']:
    assert h(base/row['staged_relative'])==row['sha256'],row['staged_relative']
original=runner.read_json(base/'control/host-native-baselines.json',transport['original_baseline_witness_sha256'])
witness=json.loads(json.dumps(original));mapping=transport['native_witness_path_rebindings']
witness['provenance']['sources_sha256']={str(base/mapping[path]):pin for path,pin in original['provenance']['sources_sha256'].items()}
for prefix in ('cpu_binary','terminal_oracle','terminal_oracle_build'):
    witness['provenance'][prefix+'_path']=str(base/mapping[original['provenance'][prefix+'_path']])
assert witness['splits']==original['splits']
witness_path=base/'control/runtime-native-baselines.json'
if witness_path.exists():
    assert runner.read_json(witness_path)==witness
    witness_pin=h(witness_path)
else:
    witness_pin=runner.publish_json(witness_path,witness)
original_kernel=runner.read_json(base/'control/host-kernel-proposed.json')
config_record=original_kernel['config'].copy();config_record['seeds']=tuple(config_record['seeds'])
config=study.Configuration(**config_record)
main=base/'main';packs=main/'results/semantic-corpus/packet-v2'
splits={split:dict(path=str(packs/split),manifest_sha256=h(packs/split/'manifest.json'),counts=runner.read_json(packs/split/'manifest.json')['counts'])for split in ('train','validation','test')}
runner.validate_witness(witness,splits)
dependencies=study.dependency_source_pins(main)
kernel=study.proposal(train_manifest_sha256=splits['train']['manifest_sha256'],validation_manifest_sha256=splits['validation']['manifest_sha256'],dependencies=dependencies,config=config)
kernel['study_source_sha256']=h(study.__file__)
# The protocol must be byte-value identical after removing only path bindings.
a=dict(original_kernel);a.pop('dependencies');b=dict(kernel);b.pop('dependencies');assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
kernel_path=base/'control/kernel-proposed.json';kernel_pin=runner.publish_json(kernel_path,kernel)
suite=runner.suite_proposal(main,packs,kernel_path,witness_path,witness_pin,base/'attempt-1')
suite_path=base/'control/suite-proposed-prekernel-seal.json';suite_pin=runner.publish_json(suite_path,suite)
review=dict(kind='semantic_runtime_metadata_path_rebinding_review_v1',base=str(base),payloads_verified=len(transport['sources']),original_witness_sha256=transport['original_baseline_witness_sha256'],runtime_witness_sha256=witness_pin,witness_splits_byte_value_unchanged=True,native_provenance_rebinding={old:str(base/relative)for old,relative in mapping.items()},kernel_protocol_except_dependencies_identical=True,kernel_proposal_sha256=kernel_pin,suite_preseal_proposal_sha256=suite_pin,dependencies=dependencies,runner_source_sha256=h(runner.__file__),config=original_kernel['config'],splits=splits,actual_training=False,heldout_quality_metrics=False,gpu_runtime_calls=0)
print(json.dumps(review,sort_keys=True,indent=2))
runner.publish_json(base/'control/runtime-metadata-review.json',review)
