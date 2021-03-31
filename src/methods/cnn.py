from collections import namedtuple
import torch
import math
from .defs import NONLINEARITIES


TimeDerivative = namedtuple("TimeDerivative", ["dq_dt", "dp_dt"])
StepPrediction = namedtuple("StepPrediction", ["q", "p"])
LayerDef = namedtuple("LayerDef", ["kernel_size", "in_chans", "out_chans"])


class CNN(torch.nn.Module):
    def __init__(self, layer_defs,
                 nonlinearity=torch.nn.ReLU, predict_type="deriv"):
        super().__init__()
        layers = []
        assert layer_defs[0].in_chans == layer_defs[-1].out_chans
        for layer_def in layer_defs:
            kern_size = layer_def.kernel_size
            pad = (kern_size - 1) // 2
            layers.append(torch.nn.Conv1d(
                layer_def.in_chans,
                layer_def.out_chans,
                kern_size,
                padding=pad,
            ))
            layers.append(nonlinearity())
        # Remove final nonlinearity
        layers = layers[:-1]
        self.ops = torch.nn.Sequential(*layers)
        self.predict_type = predict_type

    def forward(self, q, p):
        # Concatenate input
        # Pass through operations
        # Split input
        x = torch.cat((q, p), dim=-2)
        split_size = q.shape[-2]
        y = self.ops(x)

        if self.predict_type == "deriv":
            dq, dp = torch.split(y, [split_size, split_size], dim=-2)
            result = TimeDerivative(dq_dt=dq, dp_dt=dp)
        elif self.predict_type == "step":
            q, p = torch.split(y, [split_size, split_size], dim=-2)
            result = StepPrediction(q=q, p=p)
        else:
            raise ValueError(f"Invalid predict type {self.predict_type}")

        return result


def build_network(arch_args, predict_type):
    nonlinearity = NONLINEARITIES[arch_args.get("nonlinearity", "relu")]
    layer_defs = []
    for record in arch_args["layer_defs"]:
        layer_def = LayerDef(
            kernel_size=record["kernel_size"],
            in_chans=record["in_chans"],
            out_chans=record["out_chans"],
        )
        layer_defs.append(layer_def)
    cnn = CNN(layer_defs=layer_defs,
              nonlinearity=nonlinearity,
              predict_type=predict_type)
    return cnn
