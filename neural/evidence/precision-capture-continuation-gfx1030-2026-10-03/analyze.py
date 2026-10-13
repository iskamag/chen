import hashlib,json,math
from pathlib import Path
root=Path.cwd();work=root/'results/precision-capture-continuation-2026-10-03';partial=root/'neural/evidence/precision-capture-partial-gfx1030-2026-10-03';complete_path=work/'terminal/attempt-1/complete.json';complete=json.loads(complete_path.read_text());record=complete['records'];square=partial/'attempt-1/anchor-square-20261021.json';record={'square/20261021':json.loads(square.read_text()),**record}
assert len(record)==8 and len(complete['records'])==7
summary=dict(kind='chen_combined_precision_capture_analysis_v1',status='completed_representative_comparison_across_two_bound_attempts',attempts=dict(parent=dict(declaration_sha256='ae2939e606fed6c2d93968ab113bdb7870a617da199eaaf85215932399e4cd6f',status='terminal_failure_one_completed_square_anchor',square_sha256=hashlib.sha256(square.read_bytes()).hexdigest()),continuation=dict(declaration_sha256=complete['declaration_sha256'],status='terminal_success_seven_remaining_keys',complete_sha256=hashlib.sha256(complete_path.read_bytes()).hexdigest(),elapsed_seconds=complete['elapsed_seconds'])),scope='eight structural one-seed representatives;TRAIN numerical fidelity and eager/graph timings;no retraining/calibration/heldout quality/promotion/engine or rating claim',fidelity={},timings={},capture_checks=dict(graphs=0,checks=0,changed_features_checks=0,changed_actions_checks=0,all_exact=True),cold_setup_wall_ms=[])
for phase in ('anchor','transition'):
 rows=[r for r in record.values()if r['phase']==phase];summary['fidelity'][phase]={}
 for variant in rows[0]['fidelity']:
  v=[r['fidelity'][variant]for r in rows];gaps=[g for x in v for g in x['changed_fp32_best_second_gaps_cp']]
  summary['fidelity'][phase][variant]=dict(models=len(rows),cp_mean_absolute_range=[min(x['cp']['mean_absolute']for x in v),max(x['cp']['mean_absolute']for x in v)],cp_maximum_absolute=max(x['cp']['maximum_absolute']for x in v),latent_maximum_absolute=max(x['latent']['maximum_absolute']for x in v),latent_mean_absolute_range=[min(x['latent']['mean_absolute']for x in v),max(x['latent']['mean_absolute']for x in v)],roots=sum(x['roots']for x in v),changed_roots=sum(x['changed_roots']for x in v),maximum_changed_fp32_top_two_gap_cp=max(gaps,default=0),first_canonical_action_agreement=None if not sum(x['roots']for x in v)else 1-sum(x['changed_roots']for x in v)/sum(x['roots']for x in v))
for key,row in record.items():
 for shape,t in row['timings'].items():
  summary['timings'][key+'/'+shape]=t['medians']
  for v,z in t['setup'].items():
   summary['capture_checks']['graphs']+=1;summary['cold_setup_wall_ms'].append(z['synchronized_wall_ms'])
   assert len(z['numerical_checks'])==2
   for c in z['numerical_checks']:
    summary['capture_checks']['checks']+=1
    summary['capture_checks']['changed_features_checks']+=c['input_features_changed'];summary['capture_checks']['changed_actions_checks']+=c['input_actions_changed']
    for field in ('value_cp','latent'):assert c[field]['maximum_absolute']==0
summary['cold_setup_wall_ms_range']=[min(summary['cold_setup_wall_ms']),max(summary['cold_setup_wall_ms'])];del summary['cold_setup_wall_ms']
summary['cold_setup_scope']='3warmups+synchronization+capture;exact own-eager input checks happen after reported setup time;not compile-only or cold production startup'
summary['median_ratios']={}
for phase,shapes in [('anchor',['1','32','256']),('transition',['complete_root'])]:
 for shape in shapes:
  rows=[r['timings'][shape]['medians']for r in record.values()if r['phase']==phase]
  summary['median_ratios'][phase+'/'+shape]={v:dict(fp32_graph_over_variant_graph_range=[min(z['fp32_value']['graph_replay']['synchronized_wall_us']/z[v]['graph_replay']['synchronized_wall_us']for z in rows),max(z['fp32_value']['graph_replay']['synchronized_wall_us']/z[v]['graph_replay']['synchronized_wall_us']for z in rows)],same_variant_eager_over_graph_range=[min(z[v]['eager']['synchronized_wall_us']/z[v]['graph_replay']['synchronized_wall_us']for z in rows),max(z[v]['eager']['synchronized_wall_us']/z[v]['graph_replay']['synchronized_wall_us']for z in rows)])for v in rows[0]}
summary['conclusion']='Capture reduces fixed dispatch costs strongly at small batches. INT8 replay beats FP32 for all four batch1 anchors, loses for all batch32/256 anchors, and is near tied for conditioned transitions. Native/Torch FP16 and autocast do not beat FP32 replay in these models/shapes. No blanket precision promotion; preserve batch-specific variants for future producer+search integration.'
(work/'analysis.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(capture_checks=summary['capture_checks'],cold=summary['cold_setup_wall_ms_range'],fidelity=summary['fidelity'],ratios=summary['median_ratios']),indent=2))
