import hashlib,json,statistics
from pathlib import Path
from neural.strategic_transport import load_strategic_checkpoint
report_path=Path('neural/evidence/strategic-study-gfx1030-2026-10-01.json')
r=json.loads(report_path.read_text());decl_path=Path('neural/evidence/strategic-study-predeclared-2026-10-01.json');decl=json.loads(decl_path.read_text())
assert len(r['runs'])==30 and all('after_test' in x for x in r['runs'])
assert r['status']=='completed_once_after_sealed_selection'
assert hashlib.sha256(decl_path.read_bytes()).hexdigest()==r['predeclaration_sha256']
for p,h in decl['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
selection=Path('neural/evidence/strategic-study-gfx1030-2026-10-01-selection.json')
assert hashlib.sha256(selection.read_bytes()).hexdigest()==r['selection']['sha256']
checkpoints=[]
for row in r['runs']:
    p=Path(row['checkpoint']['path']);digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==row['checkpoint']['sha256']
    model,meta=load_strategic_checkpoint(p,device='cpu')
    assert meta['provenance']['configuration']==r['configuration']
    assert meta['provenance']['fixed_target_state_sha256']==row['fixed_target_state_sha256']
    assert meta['provenance']['split_sha256']==r['split']['source_sha256']
    checkpoints.append({'path':str(p),'sha256':digest,'bytes':p.stat().st_size,'strict_cpu_loader_passed':True,'architecture':row['architecture'],'control':row['control'],'seed':row['seed']})
summary=[]
for arch in r['configuration']['architectures']:
    for control in r['configuration']['controls']:
        rows=[x for x in r['runs'] if x['architecture']==arch and x['control']==control]
        values={'architecture':arch,'control':control,'seeds':[x['seed'] for x in rows]}
        for phase in ['after_training','after_validation','after_test']:
            values[phase]={}
            for name,path in [('cp_mae',('network','ordinary_centipawn_mae')),('regret_cp',('network','ordering','mean_selected_child_regret_cp')),('same_material_direction',('same_material_teacher_gap','score_direction_concordance')),('same_material_gap_rmse_cp',('same_material_teacher_gap','score_gap_rmse_cp')),('shuffled_gap_rmse_cp',('same_material_teacher_gap','shuffled_residual_score_gap_rmse_cp')),('latent_separation_ratio',('same_material_teacher_gap','latent_action_separation_ratio')),('full_refresh_cp_mae',('full_refresh_diagnostic','ordinary_centipawn_mae'))]:
                pair=[]
                for row in rows:
                    value=row[phase]
                    for key in path:value=value[key]
                    pair.append(value)
                values[phase][name]={'mean':statistics.mean(pair),'seeds':pair,'population_sd':statistics.pstdev(pair)}
        summary.append(values)
result={'kind':'strategic_study_artifact_audit_and_frozen_summary','strength_evidence':False,'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'predeclaration_sha256':r['predeclaration_sha256'],'selection_sha256':r['selection']['sha256'],'selected':r['selection']['record']['selected'],'all_source_sha256_match':True,'all30_strict_cpu_checkpoint_loads_pass':True,'checkpoints':checkpoints,'means':summary,'baseline':{phase:{k: {'cp_mae':r['runs'][0][phase][k]['ordinary_centipawn_mae'],'regret_cp':r['runs'][0][phase][k]['ordering']['mean_selected_child_regret_cp']} for k in ['material','static_tapered']} for phase in ['after_training','after_validation','after_test']},'total_measured_training_seconds':sum(x['training_seconds'] for x in r['runs']),'ranking_only_validation_candidates':[x for x in r['selection']['record']['candidates'] if x['validation_root_regret_mean']<x['regret_gate'] and not x['eligible']],'plan_action_blind_raw_float_ordering_caveat':'CPU semantic ties exact; original GPU informative-pair gaps are microCP and cause/location unverified; numerical-sensitive ranking is not learned action reasoning'}
Path('neural/evidence/strategic-study-artifacts-2026-10-01.json').write_text(json.dumps(result,indent=2)+'\n')
print('30strictCPULoadsPASS; frozen source/selection/report hashes verified')
print('ranking-onlyvalidation',[(x['architecture'],x['control']) for x in result['ranking_only_validation_candidates']])
