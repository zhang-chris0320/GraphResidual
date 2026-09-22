import torch
from torch import nn
SEQ_TYPES=("mutation","protein","ptm")
ALL_TYPES=("disease","mutation","protein","ptm","pathway")
class MLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(64,256),nn.ReLU(),nn.Linear(256,640))
 def forward(self,x):return self.net(x)
class ResidualV2(nn.Module):
 def __init__(self,relations,beta_init=0.0):
  super().__init__();self.residual=nn.ModuleDict({k:MLP() for k in SEQ_TYPES});self.fallback=nn.ModuleDict({k:MLP() for k in SEQ_TYPES});self.project=nn.ModuleDict({k:MLP() for k in ("disease","pathway")});self.norm=nn.ModuleDict({k:nn.LayerNorm(640) for k in ALL_TYPES});self.beta=nn.Parameter(torch.tensor(float(beta_init)));self.relation_vectors=nn.Parameter(torch.empty(len(relations),640));nn.init.xavier_uniform_(self.relation_vectors);self.relations=list(relations)
 def encode(self,node_type,graph,esm=None,valid=None):
  if node_type in SEQ_TYPES:
   r=self.residual[node_type](graph)
   if esm is None:raise ValueError("ESM required")
   z=self.norm[node_type](esm+self.beta*r);fb=self.norm[node_type](self.fallback[node_type](graph))
   if valid is not None:z=torch.where(valid[:,None],z,fb)
   preserve_mask=valid if valid is not None else torch.ones(len(z),dtype=torch.bool,device=z.device)
   return z,r,preserve_mask
  z=self.norm[node_type](self.project[node_type](graph));return z,z,torch.zeros(len(z),dtype=torch.bool,device=z.device)
 def score(self,h,t,relation_index):return (h*self.relation_vectors[relation_index]*t).sum(-1)/(640**0.5)
