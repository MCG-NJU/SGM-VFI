from functools import partial
import torch.nn as nn

from model import feature_extractor
from model import flow_estimation_local

'''==========Model config=========='''


def init_model_config(F=32, W=7, depth=[2, 2, 2, 4]):
    '''This function should not be modified'''
    return {
        'embed_dims': [F, 2 * F, 4 * F, 8 * F],
        'num_heads': [8 * F // 32],
        'mlp_ratios': [4],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': depth,
        'window_sizes': [W]
    }, {
        'embed_dims': [F, 2 * F, 4 * F, 8 * F],
        'depths': depth,
        'scales': [8],
        'hidden_dims': [4 * F],
        'c': F
    }


MODEL_CONFIG = {
    'LOGNAME': 'ours-local-small',
    'MODEL_TYPE': (feature_extractor, flow_estimation_local),
    'MODEL_ARCH': init_model_config(
        F=16,
        W=7,
        depth=[2, 2, 2, 4]
    )
}