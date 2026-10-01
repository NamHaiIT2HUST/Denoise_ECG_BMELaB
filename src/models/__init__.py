from src.models.dw_cnn import DW_CNN
from src.models.dw_se import DW_SE
from src.models.dnn_dan import DNN_DAN
from src.models.fcn import FCN
from src.models.deepfilter import DeepFilter
from src.models.haar_sym_lite import HaarSymLite
from src.models.liwave import LiWave


def _normalize_model_name(name):
    return name.strip().lower().replace("-", "_")


def build_model(name, wavelet="haar", se_reduction=8, base=32, expansion=4, mid_depth=3):
    model_name = _normalize_model_name(name)
    if model_name == "dw_cnn":
        return DW_CNN(wavelet=wavelet)
    if model_name == "dw_se":
        return DW_SE(wavelet=wavelet, se_r=se_reduction)
    if model_name == "dnn_dan":
        return DNN_DAN()
    if model_name == "fcn":
        return FCN()
    if model_name == "deepfilter":
        return DeepFilter()
    if model_name == "liwave":
        return LiWave(se_reduction=se_reduction, wavelet=wavelet)
    if model_name in {"haar_sym_lite", "haarsym_lite", "hslite"}:
        return HaarSymLite(
            base=base,
            expansion=expansion,
            mid_depth=mid_depth,
            se_reduction=se_reduction,
            wavelet=wavelet,
        )
    raise ValueError(f"Unknown model: {name}")
