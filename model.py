import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerPropLinear(nn.Module):
    """Powerpropagation Linear module."""
    def __init__(self, alpha, in_features, out_features):
        super(PowerPropLinear, self).__init__()
        self.alpha = alpha
        self.w = torch.nn.Parameter(torch.rand(out_features, in_features))
        self.b = torch.nn.Parameter(torch.rand(out_features))

    def get_weights(self):
        weights = self.w.detach()
        return torch.sign(weights) * torch.pow(torch.abs(weights), self.alpha)

    def forward(self, inputs, mask=None):
        weights = self.w * torch.pow(torch.abs(self.w), self.alpha - 1.)
        if mask is not None:
            weights *= mask
        return F.linear(inputs, weights, self.b)


class MLP(nn.Module):
    """A multi-layer perceptron module."""
    def __init__(self, alpha, input_sizes=(784, 1000, 1000, 500, 100), output_sizes=(1000, 1000, 500, 100, 62)):
        super(MLP, self).__init__()
        self.alpha = alpha
        self.flatten = nn.Flatten()
        self._layers = nn.ModuleList()
        for in_features, out_features in zip(input_sizes, output_sizes):
            self._layers.append(PowerPropLinear(alpha=alpha,in_features=in_features,out_features=out_features))

    def get_weights(self):
        return [layer.get_weights() for layer in self._layers]

    def forward(self, inputs, masks=None):
        num_layers = len(self._layers)
        inputs = self.flatten(inputs)  # flatten all dimensions except batch
        for i, layer in enumerate(self._layers):
            if masks is not None:
                inputs = layer(inputs, masks[i])
            else:
                inputs = layer(inputs)
                
            if i < (num_layers - 1):
                inputs = F.relu(inputs)

        return inputs

    def loss(self, inputs, targets, masks=None):
        with torch.no_grad():
            outputs = self.forward(inputs, masks)
            dist = torch.distributions.categorical.Categorical(logits=outputs)
            loss = -torch.mean(dist.log_prob(targets))

        accuracy = torch.sum(targets == torch.argmax(outputs, dim=1)) / targets.shape[0]

        return {'loss': loss, 'acc': accuracy}
