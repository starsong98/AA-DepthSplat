from .encoder_depthsplat import *
from .common.gaussian_adapter_lpf import GaussianAdapterLPF

# the sole purpose of this model is to test out loading & implementation of new model variants
class EncoderDepthSplatLPF(EncoderDepthSplat):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        print("Building DepthSplat w/ 3D Gaussian LPF adapter from mip-splatting")
        super().__init__(cfg)

        # replace gaussian adapter with 3D LPF version
        self.gaussian_adapter = GaussianAdapterLPF(cfg.gaussian_adapter)
