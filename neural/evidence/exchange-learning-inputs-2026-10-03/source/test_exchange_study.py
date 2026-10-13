"""Focused task/split/plateau mechanics, not learned quality or throughput."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import chess
import torch

from neural.exchange_model import ExchangeModel,admit_destination_queries,task_loss
from neural.exchange_study_data import Ledger,admit_family,label_split,pack_split
from neural.exchange_study import Plateau,default_protocol,collate,evaluate,load_pack,source_closure,next_fit
from neural.tactical_inputs import ReplaySpec,compact_from_replay,to_tactical_batch,tactical_config
from neural.tests.test_tactical_operator import NoHostScalar

class ExchangeStudyMechanics(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(20261009)

    def graph(self):
        state=compact_from_replay(ReplaySpec(moves=('e2e4','d7d5')))
        return state,to_tactical_batch((state,))

    def test_zero_correction_is_immediate_baseline_and_no_future_input(self):
        state,batch=self.graph();model=ExchangeModel(tactical_config(width=8))
        parameters=sum(p.numel() for p in model.parameters())
        self.assertEqual(parameters,model.estimate(model.settings)['parameters'])
        for steps in (1,2,4,8):
            with NoHostScalar():out=model(batch,steps=steps)
            torch.testing.assert_close(out['restricted_return_cp'],batch.action_prior_cp-batch.value_prior_cp[:,None],rtol=0,atol=0)
            self.assertFalse(out['strategic_value'])
        self.assertEqual(ExchangeModel.estimate(tactical_config(width=256))['parameters'],1067521)
        with self.assertRaises(ValueError):ExchangeModel(tactical_config(width=256),max_parameters=1000000)

    def test_masked_loss_and_processor_gradients(self):
        state,batch=self.graph();model=ExchangeModel(tactical_config(width=8))
        mask=torch.zeros_like(batch.action_mask);mask[0,next(i for i,a in enumerate(state.actions) if a.capture)]=True
        target=torch.full(mask.shape,float('nan'));target[mask]=-100.
        loss=task_loss(model(batch),target,mask);self.assertTrue(torch.isfinite(loss))
        loss.backward();self.assertGreater(model.task_head.weight.grad.abs().sum().item(),0)
        # Zero-init readout blocks processor gradient only on the first update.
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001);optimizer.step();optimizer.zero_grad()
        task_loss(model(batch),target,mask).backward()
        self.assertGreater(model.cell.source.weight.grad.abs().sum().item(),0)
        with self.assertRaises(ValueError):task_loss({'correction_cp':torch.zeros(mask.shape)},target,mask)

    def test_same_processor_controls_and_exact_child_query(self):
        state,batch=self.graph();model=ExchangeModel(tactical_config(width=8))
        torch.nn.init.normal_(model.task_head.weight,std=.01)
        self.assertFalse(torch.equal(model(batch,typed=True)['correction_cp'],model(batch,typed=False)['correction_cp']))
        moves=('e2e4','d7d5','e4d5');child=compact_from_replay(ReplaySpec(moves=moves));child_batch=to_tactical_batch((child,))
        query=torch.tensor([chess.D5^56]);admit_destination_queries(child_batch,query)
        with NoHostScalar():value=model.destination_continuation(child_batch,query)
        self.assertEqual(value.shape,(1,))
        for bad in (torch.tensor([-1]),torch.tensor([64]),torch.tensor([chess.A4])):
            with self.assertRaises(ValueError):admit_destination_queries(child_batch,bad)

    def test_whole_family_freshness_includes_all_children(self):
        ledger=Ledger(lambda board:board.fen());occupied=dict(keys=set(),families=set(),moves=set())
        row=dict(family_id='f',moves=['e2e4'],roots=[dict(feature_key='a',actions=[dict(child_feature_key='b')])])
        ledger.keys.add('b')
        self.assertEqual(admit_family(row,ledger,occupied)[0],'spent_feature')
        self.assertFalse(occupied['keys']);ledger.keys.clear()
        self.assertIsNone(admit_family(row,ledger,occupied)[0])
        other={**row,'family_id':'g','moves':['d2d4']}
        self.assertEqual(admit_family(other,ledger,occupied)[0],'cross_family_feature')

    def test_test_gate_rejects_before_any_artifact_open(self):
        with patch('neural.exchange_study_data.read_json',side_effect=AssertionError('opened artifact')):
            for call in (lambda:label_split('/missing','test',None),lambda:pack_split('/missing','test')):
                with self.assertRaisesRegex(ValueError,'TEST gate CLOSED'):call()
        with patch('torch.load',side_effect=AssertionError('opened TEST pack')):
            with self.assertRaisesRegex(ValueError,'TEST gate CLOSED'):load_pack('/missing','test')

    def test_plateau_requires_minimum_epochs_and_two_lr_continuations(self):
        protocol=default_protocol();plateau=Plateau(protocol)
        self.assertEqual(plateau.update(50.,1),'continue')
        for epoch in range(2,20):self.assertEqual(plateau.update(50.,epoch),'continue')
        self.assertFalse(plateau.converged)
        self.assertEqual(plateau.update(50.,20),'reduce_lr')
        for epoch in range(21,28):self.assertEqual(plateau.update(50.,epoch),'continue')
        self.assertEqual(plateau.update(50.,28),'reduce_lr')
        for epoch in range(29,36):self.assertEqual(plateau.update(50.,epoch),'continue')
        self.assertEqual(plateau.update(50.,36),'plateau')
        self.assertTrue(plateau.converged)
        with self.assertRaises(ValueError):plateau.update(float('nan'),37)

    def test_active_capacity_collation_and_literal_immediate_control(self):
        from dataclasses import fields
        import time
        records=[]
        for i,moves in enumerate((('e2e4','d7d5'),('d2d4','e7e5'))):
            state=compact_from_replay(ReplaySpec(moves=moves));batch=to_tactical_batch((state,))
            k=len(state.actions);mask=torch.tensor([a.capture for a in state.actions]);target=torch.zeros(k);target[mask]=-100.
            records.append(dict(observation={f.name:getattr(batch,f.name) for f in fields(batch)},
                correction=target,mask=mask,rounds=torch.ones(k,dtype=torch.int64),family_id=str(i),motifs=['capture']))
        batch,target,mask,rounds,meta=collate(records)
        self.assertEqual(batch.edge_index.shape[1],max(r['observation']['edge_index'].shape[1] for r in records))
        self.assertLess(batch.edge_index.shape[1],6144)
        self.assertLessEqual(meta['active_edges'],meta['capacity_edges'])
        report=evaluate(None,records,default_protocol(), 'cpu',1,True,time.monotonic()+10)
        self.assertEqual(report['macro_family_mae_cp'],100.)
        self.assertEqual(report['macro_nontrivial_family_mae_cp'],100.)
        closure=source_closure(('exchange_study.py','int8_inference.py'))
        self.assertIn('exchange_model.py',closure);self.assertIn('models.py',closure)

    def test_fit_groups_reuse_screen_seed_and_refuse_unconverged_selection(self):
        p=default_protocol();completed={}
        first,_=next_fit(p,completed);self.assertEqual(first['seed'],6101)
        for score in (100.,90.,80.,70.):
            job,_=next_fit(p,completed)
            completed[job['name']]={'report':dict(converged=True,best_dev_macro_mae_cp=score,
                learning_rate=job['lr'],weight_decay=job['wd'],seed=job['seed'])}
        job,selected=next_fit(p,completed)
        self.assertEqual(job['seed'],6102);self.assertEqual(selected['r1-typed']['reused_seed'],6101)
        completed[job['name']]={'report':{'converged':False}}
        self.assertEqual(next_fit(p,completed)[0]['seed'],6103)
        completed[next_fit(p,completed)[0]['name']]={'report':{'converged':False}}
        self.assertFalse(next_fit(p,completed)[0]['typed'])
        # One safety-stopped HPO fit invalidates hyperparameter selection for the
        # entire arm, rather than favoring a faster converging competitor.
        del completed['r1-typed-selected-seed6102'];del completed['r1-typed-selected-seed6103']
        completed[first['name']]['report']['converged']=False
        job,selected=next_fit(p,completed)
        self.assertFalse(job['typed']);self.assertTrue(selected['r1-typed']['status'].startswith('no_selection'))

if __name__=='__main__':unittest.main()
