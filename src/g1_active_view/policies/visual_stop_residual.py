from __future__ import annotations
import torch
from torch import nn
from g1_active_view.actions import ACTIONS, encode_history
from g1_active_view.vision.dinov2_encoder import DINOv2ViTB14, FEATURE_DIM, image_tensor
from g1_active_view.vision.sift_geometry import geometry, FLOW_INDICES

class MagnitudePolicy(nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('geo_mean', torch.zeros(len(FLOW_INDICES))); self.register_buffer('geo_std', torch.ones(len(FLOW_INDICES)))
        self.geo=nn.Sequential(nn.Linear(len(FLOW_INDICES),64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,48),nn.GELU()); self.visual=nn.Sequential(nn.Linear(FEATURE_DIM,48,bias=False),nn.Tanh()); self.embed=nn.Embedding(10,12,padding_idx=9); self.hist=nn.GRU(12,24,batch_first=True); self.head=nn.Sequential(nn.Linear(168,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,9))
    def forward(self,g,c,t,h):
        z=self.geo(((g[:,FLOW_INDICES]-self.geo_mean)/self.geo_std).clamp(-8,8)); out,_=self.hist(self.embed(h)); length=(h!=9).sum(1); hist=out[torch.arange(len(h),device=h.device),(length-1).clamp(min=0)]*(length>0)[:,None]
        if self.training: hist=hist*(torch.rand((len(h),1),device=h.device)>.5)
        return self.head(torch.cat([z,self.visual(t-c),self.visual((t-c).abs()),hist],1))

class MonotoneStopPolicy(nn.Module):
    def __init__(self):
        super().__init__(); self.movement=MagnitudePolicy(); self.movement.requires_grad_(False); self.register_buffer('distance_scale',torch.ones(len(FLOW_INDICES)+2)); self.raw_weights=nn.Parameter(torch.full((len(FLOW_INDICES)+2,),-3.)); self.raw_bias=nn.Parameter(torch.tensor(0.))
    def train(self, mode=True): super().train(mode); self.movement.eval(); return self
    def distances(self,g,c,t):
        delta=t-c; return torch.cat([g[:,FLOW_INDICES].abs(),delta.square().mean(1,keepdim=True).sqrt(),delta.abs().mean(1,keepdim=True)],1)
    def forward(self,g,c,t,h):
        self.movement.eval(); move=self.movement(g,c,t,h)[:,:8]; move=move-move.max(1,keepdim=True).values; d=self.distances(g,c,t)/self.distance_scale; stop=torch.nn.functional.softplus(self.raw_bias)-(d*torch.nn.functional.softplus(self.raw_weights)).sum(1); return torch.cat([move,stop[:,None]],1)

class VisualStopResidualPolicy(MonotoneStopPolicy):
    def __init__(self):
        super().__init__(); self.stop_residual=nn.Sequential(nn.Linear(len(FLOW_INDICES)+2+24,32),nn.GELU(),nn.Linear(32,1)); nn.init.zeros_(self.stop_residual[-1].weight); nn.init.zeros_(self.stop_residual[-1].bias)
    def forward(self,g,c,t,h):
        logits=super().forward(g,c,t,h); signed=((g[:,FLOW_INDICES]-self.movement.geo_mean)/self.movement.geo_std).clamp(-8,8); distance=self.distances(g,c,t)/self.distance_scale; hidden,_=self.movement.hist(self.movement.embed(h)); length=(h!=9).sum(1); history=hidden[torch.arange(len(h),device=h.device),(length-1).clamp(min=0)]*(length>0)[:,None]; residual=self.stop_residual(torch.cat([signed,distance[:,-2:],history],1))[:,0]; gate=torch.tanh(10*distance[:,-1]); return torch.cat([logits[:,:8],(logits[:,8]+gate*residual)[:,None]],1)

class OfflinePolicy:
    def __init__(self, checkpoint, dino_source, dino_weights, device='cpu'):
        self.device=torch.device(device); saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved.get('actions') != list(ACTIONS): raise ValueError(f"checkpoint action table mismatch: {saved.get('actions')}")
        if saved.get('variant') != 'visual_stop_residual': raise ValueError(f"unsupported checkpoint variant: {saved.get('variant')}")
        self.policy=VisualStopResidualPolicy().to(self.device).eval(); self.policy.load_state_dict(saved['policy']); self.encoder=DINOv2ViTB14(dino_source,dino_weights).to(self.device).eval(); self.cache={}
    @torch.inference_mode()
    def predict(self,current_rgb,target_rgb,action_history):
        g=torch.from_numpy(geometry(current_rgb,target_rgb))[None].to(self.device); c=self.encoder(image_tensor(current_rgb)[None].to(self.device)); key=str(target_rgb) if isinstance(target_rgb,(str,bytes,__import__('pathlib').Path)) else id(target_rgb)
        if key not in self.cache: self.cache[key]=self.encoder(image_tensor(target_rgb)[None].to(self.device))
        t=self.cache[key]; z=self.policy(g,c,t,torch.tensor([encode_history(action_history)],dtype=torch.long,device=self.device))[0]; i=int(z.argmax()); sim=float(torch.nn.functional.cosine_similarity(c,t,dim=1)[0].clamp(-1,1)); return {'logits':z.cpu().tolist(),'probabilities':z.softmax(-1).cpu().tolist(),'action_id':i,'action_name':ACTIONS[i],'similarity':sim,'visual_inliers':int(round(float(g[0,1])*500))}
