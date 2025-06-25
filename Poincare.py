import sys
sys.path.append("/content/poincare-embeddings")


import torch as th
from torch.autograd import Function
from hype.manifolds.euclidean import EuclideanManifold
import numpy as np

class PoincareManifold(EuclideanManifold):
    def __init__(self, eps=1e-5, K=None, **kwargs):
        self.eps = eps
        super(PoincareManifold, self).__init__(max_norm=1 - eps)
        self.K = K
        if K is not None:
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * self.K))

    def distance(self, u, v):
        return Distance.apply(u, v, self.eps)

    def half_aperture(self, u):
        eps = self.eps
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return th.asin((self.inner_radius * (1 - sqnu) / th.sqrt(sqnu))
                       .clamp(min=-1 + eps, max=1 - eps))

    def angle_at_u(self, u, v):
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)
        num = (dot_prod * (1 + norm_v ** 2) - norm_v ** 2 * (1 + norm_u ** 2))
        denom = (norm_v * edist * (1 + norm_v**2 * norm_u**2 - 2 * dot_prod).sqrt())
        return (num / denom).clamp(min=-1 + self.eps, max=1 - self.eps).acos()

    def rgrad(self, p, d_p):
        if d_p.is_sparse:
            p_sqnorm = th.sum(
                p[d_p._indices()[0].squeeze()] ** 2, dim=1,
                keepdim=True
            ).expand_as(d_p._values())
            n_vals = d_p._values() * ((1 - p_sqnorm) ** 2) / 4
            n_vals.renorm_(2, 0, 5)
            d_p = th.sparse.DoubleTensor(d_p._indices(), n_vals, d_p.size())
        else:
            p_sqnorm = th.sum(p ** 2, dim=-1, keepdim=True)
            d_p = d_p * ((1 - p_sqnorm) ** 2 / 4).expand_as(d_p)
        return d_p


# new cell

class Distance(torch.autograd.Function):
    @staticmethod
    def grad(u, v, eps):  # Gradient with respect to u
        squnorm = torch.sum(u ** 2, dim=-1)  # Squared norm of input u
        sqvnorm = torch.sum(v ** 2, dim=-1)  # Squared norm of comparison v
        sqdist = torch.sum((u - v) ** 2, dim=-1)  # Squared Euclidean distance
        alpha = 1 - squnorm  # Edge factor for u
        beta = 1 - sqvnorm  # Edge factor for v
        z = 1 + 2 * sqdist / (alpha * beta)  # Hyperbolic core term
        a = ((sqvnorm - 2 * torch.sum(u * v, dim=-1) + 1) / alpha ** 2)  # Gradient scalar term
        a = a.unsqueeze(-1).expand_as(u)  # Reshape for broadcasting
        a = a * u - v / alpha.unsqueeze(-1).expand_as(v)  # Full direction adjustment
        z = torch.sqrt(z ** 2 - 1)  # Arcosh sqrt term
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)  # Stabilize and reshape
        return 4 * a / z.expand_as(u)  # Final scaled gradient

    @staticmethod
    def forward(ctx, u, v, eps):  # Compute Poincaré distance
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)  # Squared norm of u
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)  # Squared norm of v
        sqdist = torch.sum((u - v) ** 2, dim=-1)  # Squared Euclidean distance
        ctx.eps = eps  # Save epsilon for backward pass
        ctx.save_for_backward(u, v)  # Save inputs for backward
        alpha = 1 - squnorm  # Edge factor A
        beta = 1 - sqvnorm  # Edge factor B
        z = 1 + 2 * sqdist / (alpha * beta)  # Core distance term
        return torch.log(z + torch.sqrt(z ** 2 - 1))  # Hyperbolic arcosh distance

    @staticmethod
    def backward(ctx, g):  # Backpropagation for distance
        u, v = ctx.saved_tensors  # Retrieve saved inputs
        g = g.unsqueeze(-1)  # Reshape gradient
        gu = Distance.grad(u, v, ctx.eps)  # Gradient wrt u
        gv = Distance.grad(v, u, ctx.eps)  # Gradient wrt v
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None  # Return gradients