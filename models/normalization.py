import torch
import torch.nn as nn

##############################################################
# In this script we define the possible normalization layers #
##############################################################

# Here we choose which normalization we want to use, as chosen in the config file
def get_normalization(config):
    norm = config.model.normalization
    if norm == 'BatchNorm':
        return nn.BatchNorm2d
    elif norm == 'InstanceNorm':
        return nn.InstanceNorm2d
    elif norm == 'InstanceNorm++':
        return InstanceNorm2dPlus
    elif norm == 'VarianceNorm':
        return VarianceNorm2d
    elif norm == 'NoneNorm':
        return NoneNorm2d
    elif norm is None:
        return None
    else:
        raise NotImplementedError("{} does not exist!".format(norm))


# Here we define all the different normalization techniques (as layers) that are not already included in PyTorch
class VarianceNorm2d(nn.Module):
    def __init__(self, num_features, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)

    def forward(self, x):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        out = self.alpha.view(-1, self.num_features, 1, 1) * h
        return out


class NoneNorm2d(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()

    def forward(self, x):
        return x


class InstanceNorm2dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)
        else:
            h = h + means[..., None, None] * self.alpha[..., None, None]
            out = self.gamma.view(-1, self.num_features, 1, 1) * h
        return out