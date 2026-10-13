"""CPU-only frozen checkpoint/root geometry diagnostic; no teacher labels."""
import hashlib,json
from pathlib import Path
import chess
import torch
from neural.features import encode_fen,move_to_index
from neural.strategic_transport import load_strategic_checkpoint
report_path=Path('neural/evidence/strategic-study-gfx1030-2026-10-01.json')
report=json.loads(report_path.read_text())
roots=[row['root'] for row in report['test_corpus']['lineage'] if 'omitted' not in row]
features=[];actions=[];sizes=[];group_records=[];offset=0
for root in roots:
    board=chess.Board(root['fen']);moves=list(board.legal_moves);count=len(moves)
    features.append(encode_fen(root['fen']).repeat(count,1,1,1))
    actions += [move_to_index(move.uci(),'w' if board.turn else 'b') for move in moves]
    sizes.append(count)
    group_records.append({'root_fen':root['fen'],'offset':offset,'rows':count,'crosses_batch128_boundary':offset//128 != (offset+count-1)//128})
    offset+=count
features=torch.cat(features);actions=torch.tensor(actions,dtype=torch.int64)
results=[]
for seed in report['configuration']['seeds']:
    path=Path(f'results/neural/strategic-study/plan-action_blind-{seed}.pt')
    model,_=load_strategic_checkpoint(path,device='cpu');model.eval()
    chunked=[];anchors=[];cached=[]
    with torch.no_grad():
        for start in range(0,offset,128):
            prediction=model.predict(features[start:start+128],actions[start:start+128])
            chunked.append(prediction['residual_cp']);anchors.append(prediction['material_cp'])
        start=0
        for count in sizes:
            # One root encoding/CP residual; broadcast independent of action.
            latent=model.encode(features[start:start+1])
            residual=model.predict(features[start:start+1],actions[start:start+1],root_latent=latent)['residual_cp']
            cached.append(residual.expand(count));start+=count
    chunked,anchors,cached=map(torch.cat,(chunked,anchors,cached))
    diagnostics=[];start=0
    for geometry,count in zip(group_records,sizes):
        residual=chunked[start:start+count];material=anchors[start:start+count]
        cp=residual+material;fixed_cp=cached[start:start+count]+material
        tied=material[:,None]==material[None,:]
        same_gap=(residual[:,None]-residual[None,:]).abs()
        fixed_gap=(fixed_cp[:,None]-fixed_cp[None,:]).abs()
        assert float(fixed_gap[tied].max())==0
        assert bool(((fixed_cp==fixed_cp.min())==(material==material.min())).all())
        diagnostics.append({**geometry,'residual_range_cp':float(residual.max()-residual.min()),
                            'same_material_max_residual_gap_cp':float(same_gap[tied].max()),
                            'same_material_ties_numerically_split':int(((same_gap>0)&tied).sum()),
                            'chunked_selected_material_tie_set_changed':not torch.equal(cp==cp.min(),material==material.min()),
                            'cached_root_material_ties_exact':True})
        start+=count
    gpu=next(x for x in report['runs'] if x['architecture']=='plan' and x['control']=='action_blind' and x['seed']==seed)['after_test']
    results.append({'seed':seed,'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'cpu_root_residual_max_range_cp':max(x['residual_range_cp'] for x in diagnostics),
                    'cpu_roots_with_nonconstant_chunked_residual':sum(x['residual_range_cp']>0 for x in diagnostics),
                    'cpu_material_tie_sets_changed':sum(x['chunked_selected_material_tie_set_changed'] for x in diagnostics),
                    'cached_root_all_material_ties_exact':True,'roots':diagnostics,
                    'original_gpu_informative_pair_mean_abs_blind_gap_cp':gpu['same_material_teacher_gap']['predicted_mean_abs_score_gap_cp'],
                    'original_gpu_raw_float_regret_cp':gpu['network']['ordering']['mean_selected_child_regret_cp'],
                    'original_gpu_material_regret_cp':gpu['material']['ordering']['mean_selected_child_regret_cp']})
out={'kind':'cpu_only_frozen_plan_actionblind_batch_geometry_diagnostic','strength_evidence':False,
     'source_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'torch_cpu':str(torch.__version__),
     'no_raw_teacher_labels_generated_or_read':True,'no_gpu_calls':True,'rows':offset,'roots':len(roots),
     'geometry_root_crossing_batch128_boundaries':sum(x['crosses_batch128_boundary'] for x in group_records),
     'action_order_scope':'python-chess enumeration; same root observation/count geometry as original study; not native chosen-move reproduction',
     'diagnostic_scope':'CPU confirms semantic material ties and did not reproduce GPU tie splitting; original GPU microCP gaps are consistent with roundoff, cause/batch-boundary location unverified; original scored study unchanged',
     'runs':results}
Path('neural/evidence/strategic-study-actionblind-roundoff-2026-10-01.json').write_text(json.dumps(out,indent=2)+'\n')
print([(x['seed'],x['cpu_root_residual_max_range_cp'],x['cpu_material_tie_sets_changed']) for x in results])
