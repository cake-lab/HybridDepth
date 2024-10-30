import torch
from model.depthNet import DepthNet

def depthnet(pretrained=False, **kwargs):
    model = DepthNet(**kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url('URL_TO_YOUR_CHECKPOINT')
        model.load_state_dict(checkpoint)
    return model