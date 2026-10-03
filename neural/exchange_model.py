"""Restricted-exchange correction task head over the shared tactical processor.

This output is physical material DELTA for an artificial recapture game, never
strategic SF CP or a general chess value. No labels/future states in forward.
"""
from dataclasses import asdict

import torch
from torch import nn

from .tactical_operator import TacticalCell, TacticalConfig, STEPS, _metadata

KIND='chen_restricted_exchange_correction_v1'
FRAME='initial_parent_mover_material_delta_cp'


class ExchangeModel(nn.Module):
    def __init__(self, config, *, max_parameters=None):
        super().__init__()
        if not isinstance(config,TacticalConfig):raise ValueError('TacticalConfig required')
        count=self.estimate(config)['parameters']
        if max_parameters is not None and (type(max_parameters) is not int or count>max_parameters):
            raise ValueError('declared parameter allocation budget exceeded')
        self.settings=config
        d=config.width
        self.piece_input=nn.Linear(config.piece_channels,d)
        self.square_input=nn.Linear(config.square_channels,d)
        self.action_input=nn.Linear(config.action_channels,d)
        self.cell=TacticalCell(d,config.edge_channels)
        self.task_head=nn.Linear(4*d,1)
        nn.init.zeros_(self.task_head.weight)
        nn.init.zeros_(self.task_head.bias)

    @staticmethod
    def estimate(config):
        # Same inputs/processor as TacticalOperator; one4D head, no strategic
        # value or action CP heads, no unused decoder parameters.
        d=config.width
        count=(config.estimate()['processor_parameters']+
               (config.piece_channels+config.square_channels+config.action_channels+7)*d+1)
        return dict(parameters=count,processor_parameters=config.estimate()['processor_parameters'],
                    stored_fp32_parameter_bytes=4*count,allocated=False,measured_latency=None)

    def config(self):
        return dict(kind=KIND,**asdict(self.settings),frame=FRAME,
                    target='closed_restricted_exchange_material_correction',
                    strategic_value=False,current_evidence_only=True)

    def encode(self,batch,*,steps=None,typed=True):
        b,p,k,_=_metadata(batch,self.settings)
        steps=self.settings.steps if steps is None else steps
        if type(steps) is not int or steps not in STEPS or type(typed) is not bool:
            raise ValueError('declared recurrence and boolean relation control required')
        if batch.pieces.device!=self.piece_input.weight.device or self.piece_input.weight.dtype!=torch.float32:
            raise ValueError('same-device FP32 task model required')
        clean=lambda x,m:torch.where(m[:,:,None],x,torch.zeros_like(x))
        pieces=self.piece_input(clean(batch.pieces,batch.piece_mask))
        squares=self.square_input(batch.squares)
        actions=self.action_input(clean(batch.actions,batch.action_mask))
        mask=torch.cat((batch.piece_mask,torch.ones(b,64,dtype=torch.bool,device=pieces.device),batch.action_mask),1)
        latent=torch.where(mask[:,:,None],torch.cat((pieces,squares,actions),1),0.)
        for _ in range(steps):latent=self.cell(latent,mask,batch,typed=typed)
        pool=lambda values,valid:values.sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        pooled=torch.cat((pool(latent[:,:p],batch.piece_mask),latent[:,p:p+64].mean(1),
                          pool(latent[:,p+64:],batch.action_mask)),1)
        return latent,pooled,p,k

    def forward(self,batch,*,steps=None,typed=True):
        latent,pooled,p,k=self.encode(batch,steps=steps,typed=typed)
        correction=self.task_head(torch.cat((latent[:,p+64:],pooled[:,None].expand(-1,k,-1)),-1)).squeeze(-1)*self.settings.cp_scale
        correction=correction.masked_fill(~batch.action_mask,0.)
        immediate=(batch.action_prior_cp-batch.value_prior_cp[:,None]).masked_fill(~batch.action_mask,0.)
        return dict(correction_cp=correction,restricted_return_cp=immediate+correction,
                    immediate_material_cp=immediate,frame=FRAME,strategic_value=False)

    def destination_continuation(self,child_batch,query_square,*,steps=None,typed=True):
        """Same head/weights exact-child control: current-mover gain at target.

        Query square is caller-admitted current-child canonical occupied square.
        Full child graph is exact current observation; no future trace/target.
        Parent correction is NEGATIVE of this child-mover continuation output.
        """
        latent,pooled,p,_=self.encode(child_batch,steps=steps,typed=typed)
        if (not isinstance(query_square,torch.Tensor) or query_square.dtype!=torch.int64 or
            query_square.shape!=(latent.shape[0],) or query_square.device!=latent.device):
            raise ValueError('same-device int64 query square per child row required')
        # Range/content admission stays cold; no device scalar reads.
        square=latent[:,p:p+64].gather(1,query_square.clamp(0,63)[:,None,None].expand(-1,1,latent.shape[-1])).squeeze(1)
        return self.task_head(torch.cat((square,pooled),-1)).squeeze(-1)*self.settings.cp_scale


def task_loss(output,target_correction,mask,*,scale=800.):
    """Only closed capture/promotion slots; exact targets, no mate/bound guessing."""
    prediction=output['correction_cp']
    if (output.get('frame')!=FRAME or output.get('strategic_value') is not False or
        type(scale) not in (int,float) or not 0<scale<=10000 or
        mask.dtype!=torch.bool or prediction.shape!=mask.shape or
        target_correction.dtype!=prediction.dtype or
        target_correction.shape!=mask.shape or prediction.device!=mask.device or
        target_correction.device!=mask.device):raise ValueError('resident task target/mask shape mismatch')
    clean=torch.where(mask,target_correction,torch.zeros_like(target_correction))
    row=torch.nn.functional.smooth_l1_loss(prediction/scale,clean/scale,reduction='none')
    return torch.where(mask,row,torch.zeros_like(row)).sum()/mask.sum().clamp_min(1)


def admit_destination_queries(batch,query_square):
    """Cold reference admission; source replay/parent-child identity is external.

    A destination must be an occupied current-child square. This function does
    not certify that an arbitrary child actually follows its supplied parent.
    """
    if (batch.pieces.device.type!='cpu' or query_square.device.type!='cpu' or
        query_square.dtype!=torch.int64 or query_square.shape!=(len(batch.pieces),)):
        raise ValueError('cold CPU int64 destination queries required')
    for row,square in enumerate(query_square.tolist()):
        if not 0<=square<64 or square not in batch.piece_squares[row,batch.piece_mask[row]].tolist():
            raise ValueError('query is not an occupied current-child square')
