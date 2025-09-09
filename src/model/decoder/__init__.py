from ...dataset import DatasetCfg
from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
#from .decoder_mipsplatting_cuda import DecoderMipSplattingCUDA, DecoderMipSplattingCUDACfg
from .decoder_anysplat_cuda import DecoderSplattingCUDAAnySplat, DecoderSplattingCUDAAnySplatCfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
    "splatting_cuda_anysplat": DecoderSplattingCUDAAnySplat,
}

DecoderCfg = DecoderSplattingCUDACfg | DecoderSplattingCUDAAnySplatCfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)
