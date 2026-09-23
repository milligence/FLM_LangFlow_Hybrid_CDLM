"""Config-driven reference sampler using the repository's monotone tau functions.

NumPy/FP64 reference implementation. Do not substitute a toy tau for production.
No vocabulary-sized state allocation occurs here. Each map row carries a fresh
noise stream ID supplied by the trainer, independent from its local noise.
"""
from __future__ import annotations
from typing import Callable
import math
import numpy as np
from .control import stage_at, hard_weight


def _u(rng: np.random.Generator) -> float:
    u=float(rng.random())
    return float(np.nextafter(0.,1.)) if u==0 else u


def _draw(rng: np.random.Generator, a: float, b: float, law: str) -> float:
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise ValueError(f"Empty/invalid support [{a},{b}] for {law}")
    u=_u(rng)
    if law == 'uniform':
        return a+(b-a)*u
    if law == 'log_uniform' and a>0:
        return math.exp(math.log(a)+(math.log(b)-math.log(a))*u)
    raise ValueError(f"Unsupported law {law}")


def _components(local:dict,k:int,tau_inverse:Callable[[float],float]) -> list[dict]:
    stage=stage_at(local['stages'],k)
    if stage['law']=='uniform_tau':
        return [{'law':'uniform_tau','low':0.,'high':1.,'weight':1.}]
    out=[]
    for spec,n in zip(local['bins'],stage['quotas']):
        c=dict(spec);c['weight']=n/256
        if c.get('high')=='inverse_tau(1/128)':
            c['high']=float(tau_inverse(1/128))
        out.append(c)
    return out


def draw_local_marginal_conditioned(local:dict,k:int,r_max:float,rng:np.random.Generator,
                                    tau:Callable[[float],float],tau_inverse:Callable[[float],float])->float:
    """q_local(r | r<r_max), with retained-mass reweighting of original bins."""
    choices=[];masses=[]
    for c in _components(local,k,tau_inverse):
        if c['law']=='atom':
            if c['t'] < r_max:
                choices.append((c,None,None));masses.append(c['weight'])
            continue
        a,b=float(c['low']),float(c['high'])
        upper=min(b,float(tau(r_max))) if c['law']=='uniform_tau' else min(b,r_max)
        if upper>a:
            choices.append((c,a,upper));masses.append(c['weight']*(upper-a)/(b-a))
    if not masses or sum(masses)<=0:
        raise ValueError("Conditioned source has zero retained mass")
    ix=int(rng.choice(len(choices),p=np.asarray(masses)/sum(masses)))
    c,a,b=choices[ix]
    if c['law']=='atom':return float(c['t'])
    x=_draw(rng,a,b,'uniform')
    r=float(tau_inverse(x)) if c['law']=='uniform_tau' else x
    if not (0<=r<r_max):
        raise ValueError("LUT/inverse produced infeasible source; repair numerical implementation, not science")
    return r


def sample_local_times(config:dict,k:int,rng:np.random.Generator,
                       tau:Callable[[float],float],tau_inverse:Callable[[float],float])->np.ndarray:
    local=config['local'];stage=stage_at(local['stages'],k)
    if stage['law']=='uniform_tau':
        return np.array([tau_inverse(_u(rng)) for _ in range(256)],dtype=np.float64)
    result=[]
    comps=_components(local,k,tau_inverse)
    for c,n in zip(comps,stage['quotas']):
        for _ in range(n):
            if c['law']=='atom': r=c['t']
            else:
                v=_draw(rng,c['low'],c['high'],'uniform')
                r=tau_inverse(v) if c['law']=='uniform_tau' else v
            result.append(float(r))
    return np.asarray(result)[rng.permutation(256)]


def sample_map_pairs(config:dict,k:int,rng:np.random.Generator,
                     tau:Callable[[float],float],tau_inverse:Callable[[float],float],
                     profile:str|None=None)->list[dict]:
    stage=stage_at(config['map']['stages'],k)
    if not stage['map_batch']:return []
    cap=stage['terminal_cap'];classes=config['map']['classes'];rows=[]
    profile_spec=None
    counts=stage['counts']
    if profile is not None:
        profile_spec=config['map']['profiles'][profile]
        counts=profile_spec['counts']
    for label,count in counts.items():
        if not count:continue
        spec=classes[label];kind=spec['kind']
        if kind=='deployment':
            indices=spec['interval_indices'];dep=config['map']['deployment']
            names=spec.get('grids',list(dep['grids']))
            grids={name:dep['grids'][name] for name in names}
            interval_rows=None
            if label=='D_patch' and profile_spec is not None:
                interval_rows=profile_spec['D_patch_interval_rows']
                if len(interval_rows)!=len(indices) or sum(interval_rows)!=count:
                    raise ValueError('Invalid D_patch interval allocation')
            else:
                nper=count//(len(grids)*len(indices))
                if nper*len(grids)*len(indices)!=count:raise ValueError('Unbalanced D quota')
            for grid_name,grid in grids.items():
                for offset,idx in enumerate(indices):
                    per=interval_rows[offset] if interval_rows is not None else nper
                    for j in range(per):
                        nodes=np.asarray(grid,dtype=np.float64).copy()
                        jittered=(j+idx+k)%2==1
                        if jittered:
                            nodes[1:-1]+=rng.uniform(*dep['jitter_interior_uniform'],size=len(nodes)-2)
                        if np.any(np.diff(nodes)<=0):raise ValueError('Nonmonotone jittered grid')
                        r,s=float(nodes[idx]),float(nodes[idx+1]);eta=(s-r)/(1-r)
                        rows.append({'class':label,'r':r,'s':s,'eta':eta,'grid':grid_name,'interval':idx,'jittered':jittered,'nodes':nodes.tolist(),'source':spec.get('source','analytical_data')})
            continue
        for _ in range(count):
            if kind=='full_span':r,s=spec['r'],spec['s'];eta=(s-r)/(1-r)
            elif kind=='prefix':
                r=spec['r'];eta=_draw(rng,spec['eta_low'],min(spec['eta_high'],cap),spec['eta_law']);s=r+(1-r)*eta
            elif kind=='continuous':
                lo,hi=spec['eta_low'],spec['eta_high']
                rmax=(cap-lo)/(1-lo)
                r=draw_local_marginal_conditioned(config['local'],k,rmax,rng,tau,tau_inverse)
                eta=_draw(rng,lo,min(hi,(cap-r)/(1-r)),spec['eta_law']);s=r+(1-r)*eta
            else:raise ValueError(kind)
            rows.append({'class':label,'r':r,'s':s,'eta':eta,'grid':None,'interval':None,'jittered':False,'nodes':None,'source':'analytical_data'})
    for row in rows:
        if not (0<=row['r']<row['s']<=cap+1e-12):raise ValueError('Infeasible pair')
        row['chi']=row['eta']/(1-row['s'])
        row['hard']=(row['class'] in ('L','H') or (row['class'] in ('D','D_patch') and row['interval']==3))
        row['weight']=hard_weight(config,k) if row['hard'] else 1.
    if len(rows)!=stage['map_batch']:raise ValueError('Map quota mismatch')
    return [rows[i] for i in rng.permutation(len(rows))]
