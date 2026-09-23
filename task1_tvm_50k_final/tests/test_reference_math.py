import unittest
import torch
from reference.objectives import (f_map_loss, f_residual, p_map_terms,
                                  probability_mse, sequence_mean, f_outputs)
torch.set_num_threads(1)

class MathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dtype=torch.float64

    def test_f_stopped_velocity_gradient_equivalence(self):
        B,L,V=3,2,7
        a=torch.randn(B,L,V,dtype=self.dtype,requires_grad=True)
        b=torch.randn(B,L,V,dtype=self.dtype,requires_grad=True)
        c=torch.randn(B,L,V,dtype=self.dtype,requires_grad=True)
        eta=torch.tensor([.03,.4,.9],dtype=self.dtype)[:,None,None]
        r=torch.tensor([.2,.3,0.],dtype=self.dtype)[:,None,None]
        x=torch.randn_like(a);pt=torch.randn_like(a).softmax(-1)
        def output(e):
            p=(a+e*b).softmax(-1)
            return p+e*(c-c.mean(-1,keepdim=True))
        A,DA=torch.func.jvp(output,(eta,),(torch.ones_like(eta),))
        s=r+(1-r)*eta
        Y=(1-eta)*x+eta*A
        tangent=(-x+A+eta*DA)/(1-r)
        teacher=((pt-Y.detach())/(1-s)).detach()
        raw=(1-s)*(tangent-teacher)
        stable=f_residual(A,DA,eta,pt)
        self.assertTrue(torch.allclose(raw,stable,atol=1e-12,rtol=1e-12))
        l1=.5*raw.square().sum(-1).mean()
        l2=f_map_loss(A,DA,eta,pt)
        g1=torch.autograd.grad(l1,(a,b,c),retain_graph=True)
        g2=torch.autograd.grad(l2,(a,b,c),retain_graph=True)
        for u,v in zip(g1,g2):
            self.assertTrue(torch.allclose(u,v,atol=1e-11,rtol=1e-10))
        # The numerically equal but incorrectly stopped expression must be distinguishable.
        wrong=.5*(A+eta*(1-eta)*DA-pt).square().sum(-1).mean()
        gw=torch.autograd.grad(wrong,(a,b,c))
        self.assertGreater(sum((u-v).norm().item() for u,v in zip(g1,gw)),1e-4)

    def test_f_correction_product_rule(self):
        l=torch.randn(2,3,7,dtype=self.dtype,requires_grad=True)
        dl=torch.randn_like(l);b=torch.randn_like(l);db=torch.randn_like(l)
        e=torch.tensor([.02,.5],dtype=self.dtype)
        A,DA=f_outputs(l,dl,b,db,e)
        et=e[:,None,None]
        p=l.softmax(-1);dp=p*(dl-(p*dl).sum(-1,keepdim=True))
        expected=dp+(b-b.mean(-1,keepdim=True))+et*(db-db.mean(-1,keepdim=True))
        self.assertTrue(torch.allclose(DA,expected))
        self.assertTrue(torch.allclose(A.sum(-1),torch.ones_like(A.sum(-1)),atol=1e-12))

    def test_eta_gate_product_rule_and_parameter_gradient(self):
        w=torch.randn(2,4,dtype=self.dtype,requires_grad=True)
        v=torch.randn(4,3,dtype=self.dtype,requires_grad=True)
        r=torch.tensor([[.2],[.7]],dtype=self.dtype)
        e=torch.tensor([[.03],[.5]],dtype=self.dtype)
        def G(z):return torch.tanh(torch.cat([r,z],-1)@w)@v
        def gated(z):return z*G(z)
        y,dy=torch.func.jvp(gated,(e,),(torch.ones_like(e),))
        gg,dg=torch.func.jvp(G,(e,),(torch.ones_like(e),))
        self.assertTrue(torch.allclose(dy,gg+e*dg,atol=1e-12))
        h=1e-5
        fd=(gated(e+h)-gated(e-h))/(2*h)
        self.assertTrue(torch.allclose(dy,fd,atol=1e-8,rtol=1e-7))
        grads=torch.autograd.grad(dy.square().mean(),(w,v))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(g.norm()>0 for g in grads))
        zero=gated(torch.zeros_like(e))
        zg=torch.autograd.grad(zero.sum(),(w,v))
        self.assertTrue(all(g.abs().max()==0 for g in zg))

    def test_p_fixed_point(self):
        l=torch.randn(2,3,11,dtype=self.dtype,requires_grad=True)
        d=torch.zeros_like(l,requires_grad=True)
        out=p_map_terms(l,d,torch.tensor([.03,.8],dtype=self.dtype),l.detach())
        self.assertTrue(torch.allclose(out['log_qs'],out['log_qt'],atol=1e-12))
        self.assertLess(abs(out['kl_unweighted'].item()),1e-12)
        self.assertTrue(torch.allclose(out['q_raw'].sum(-1),torch.ones(2,3,dtype=self.dtype),atol=1e-12))
        g=torch.autograd.grad(out['loss'],(l,d))
        self.assertTrue(all(x.abs().max()<1e-12 for x in g))

    def test_p_negative_mass_and_extreme_logits(self):
        for dtype in [torch.float64,torch.float32]:
            l=torch.tensor([[[1e4,-1e4,0.,20.],[0.,0.,0.,0.]]],dtype=dtype,requires_grad=True)
            d=torch.tensor([[[50.,-50.,100.,-100.],[-100.,100.,-100.,100.]]],dtype=dtype,requires_grad=True)
            t=-l.detach()
            out=p_map_terms(l,d,torch.tensor([.5],dtype=dtype),t)
            self.assertTrue(torch.isfinite(out['loss']))
            self.assertGreater(out['negative_mass'].max().item(),0)
            self.assertTrue(torch.allclose(out['log_qs'].exp().sum(-1),torch.ones(1,2,dtype=dtype),atol=2e-5))
            self.assertTrue(torch.allclose(out['log_qt'].exp().sum(-1),torch.ones(1,2,dtype=dtype),atol=2e-5))
            g=torch.autograd.grad(out['loss'],(l,d))
            self.assertTrue(all(torch.isfinite(x).all() for x in g))
            self.assertFalse(out['log_qt'].requires_grad)

    def test_vocab_sum_and_weight_denominator(self):
        l=torch.randn(3,2,5,dtype=self.dtype);y=torch.tensor([[1,2],[2,3],[4,0]])
        one=torch.nn.functional.one_hot(y,5).to(self.dtype)
        expected=.5*(l.softmax(-1)-one).square().sum(-1).mean()
        self.assertTrue(torch.allclose(probability_mse(l,y),expected))
        token=torch.ones(3,2,dtype=self.dtype);w=torch.tensor([1.,.1,.1],dtype=self.dtype)
        self.assertAlmostEqual(sequence_mean(token,w).item(),.4)

if __name__=='__main__':unittest.main()
