from .neural_network import NNModel
from .transformer import TransformerInterp
# from .diffusion import DiffusionModel
# from .flow_matching import FlowMatchingModel
# from .transformer_diffusion import TransformerDiffusion
from .unet import UNetInterp
from .linear_model import LinearModel

MODEL_DICT = {
    "nn": NNModel,
    "transformer": TransformerInterp,
    # "diffusion": DiffusionModel,
    # "flow": FlowMatchingModel,
    # "transformer_diffusion": TransformerDiffusion,
    "unet": UNetInterp,
    "linear": LinearModel,
}
