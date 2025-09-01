from .encoder_depthsplat import *

# the sole purpose of this model is to test out loading & implementation of new model variants
class EncoderDummy(EncoderDepthSplat):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        print("Building dummy model")
        super().__init__(cfg)