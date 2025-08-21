from ...dataset import DatasetCfg
from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
#from .decoder_mipsplatting_cuda import DecoderMipSplattingCUDA, DecoderMipSplattingCUDACfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    #"mipsplatting_cuda": DecoderMipSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg #| DecoderMipSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
