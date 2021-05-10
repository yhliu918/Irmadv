
def proj_l1ball(x, epsilon=10, device="cuda:1"):
    assert epsilon > 0
    # compute the vector of absolute values
    u = x.abs()
    xshape = x.shape
    if (u.sum(dim=(1, 2, 3)) <= epsilon).all():
        # check if x is already a solution
        return x

    # y = x* epsilon/norms_l1(x)
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    y = proj_simplex(u, s=epsilon, device=device)
    # compute the solution to the original problem on v
    # pdb.set_trace()

    y = y.view(-1, xshape[1], xshape[2], xshape[3])
    y *= x.sign()
    return y

def proj_simplex(v, s=1, device="cuda:1"):
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    batch_size = v.shape[0]
    # check if we are already on the simplex
    '''
    #Not checking this as we are calling this from the previous function only
    if v.sum(dim = (1,2,3)) == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    '''
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = v.view(batch_size, 1, -1)
    n = u.shape[2]
    u, indices = torch.sort(u, descending=True)
    cssv = u.cumsum(dim=2)
    # get the number of > 0 components of the optimal solution
    vec = u * torch.arange(1, n + 1).float().to(device)
    comp = (vec > (cssv - s)).float()

    u = comp.cumsum(dim=2)
    w = (comp - 1).cumsum(dim=2)
    u = u + w
    rho = torch.argmax(u, dim=2)
    rho = rho.view(batch_size)
    c = torch.Tensor([cssv[i, 0, rho[i]] for i in range(cssv.shape[0])]).to(device)
    c = c - s
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = torch.div(c, (rho.float() + 1))
    theta = theta.view(batch_size, 1, 1, 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clamp(min=0)
    return w


def norms_l1(Z):
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]

def norms_l2(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

class zdh_additivenoise():
    def __init__(self, model, epsilon, max_iters, device, var=1., _type='l2'):
        self.epsilon = epsilon
        self.steps = max_iters
        self._type = _type
        self.model = model
        self.name = 'additivenoise'
        self.device = device
        self.var = var
        self.mode = 'repeat'  # std/repeat

    def perturb(self, original_images, labels, random_start=True):
        self.model.eval()
        epsilons = [self.epsilon]
        if self._type == 'l2':
            delta = torch.zeros_like(original_images).to(self.device)
            if self.mode == 'repeat':
                for steps in range(self.steps):
                    noise = self.var * torch.randn(original_images.shape).to(self.device)                   
                    delta = delta + noise
                    delta *= self.epsilon / norms_l2(delta)
                    delta = torch.clamp(original_images + delta, 0., 1.) - original_images
                advs = original_images + delta

        if self._type == 'l1':
            delta = torch.zeros_like(original_images).to(self.device)
            if self.mode == 'repeat':
                for steps in range(self.steps):
                    noise = self.var * torch.randn(original_images.shape).to(self.device)
                    delta = delta + noise
                    if (norms_l1(delta) > self.epsilon).any():
                        delta.data = proj_l1ball(delta, self.epsilon, self.device)
                    delta *= proj_l1ball(noise, self.epsilon, self.device)
                    delta = torch.clamp(original_images + delta, 0., 1.) - original_images

                advs = original_images + delta

        self.model.train()
        return [advs,]
