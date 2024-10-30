import torch
from model.main import DepthNetModule

# Each model function below corresponds to a different version of HybridDepth 
# trained with a specific number of focal stacks and datasets.

def HybridDepth_NYU5(pretrained=False, **kwargs):
    """
    Loads the HybridDepth model trained on the NYU Depth V2 dataset using a 5-focal stack input.
    
    Args:
        pretrained (bool): If True, loads model with pre-trained weights from URL.
        **kwargs: Additional keyword arguments for the DepthNetModule class.
    
    Returns:
        DepthNetModule: The initialized HybridDepth model with optional pre-trained weights.
    """
    model = DepthNetModule(**kwargs)
    if pretrained:
        # URL for pre-trained weights
        pretrained_resource = "https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest5-DFV-Trained.ckpt"
        # Load the checkpoint from the URL
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_resource, map_location='cpu')
        # Check if checkpoint contains a 'state_dict' key (some .ckpt files store this way)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    return model


def HybridDepth_NYU10(pretrained=False, **kwargs):
    """
    Loads the HybridDepth model trained on the NYU Depth V2 dataset using a 10-focal stack input.
    
    Args:
        pretrained (bool): If True, loads model with pre-trained weights from URL.
        **kwargs: Additional keyword arguments for the DepthNetModule class.
    
    Returns:
        DepthNetModule: The initialized HybridDepth model with optional pre-trained weights.
    """
    model = DepthNetModule(**kwargs)
    if pretrained:
        pretrained_resource = "https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NYUBest10-DFV-Trained.ckpt"
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_resource, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    return model


def HybridDepth_DDFF5(pretrained=False, **kwargs):
    """
    Loads the HybridDepth model trained on the DDFF (Depth from Focus) dataset using a 5-focal stack input.
    
    Args:
        pretrained (bool): If True, loads model with pre-trained weights from URL.
        **kwargs: Additional keyword arguments for the DepthNetModule class.
    
    Returns:
        DepthNetModule: The initialized HybridDepth model with optional pre-trained weights.
    """
    model = DepthNetModule(**kwargs)
    if pretrained:
        pretrained_resource = "https://github.com/cake-lab/HybridDepth/releases/download/v2.0/DDFF12.ckpt"
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_resource, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    return model


def HybridDepth_NYU_PretrainedDFV5(pretrained=False, **kwargs):
    """
    Loads the HybridDepth model trained on the NYU Depth V2 dataset using a 5-focal stack input, pre-trained on DFV.
    
    Args:
        pretrained (bool): If True, loads model with pre-trained weights from URL.
        **kwargs: Additional keyword arguments for the DepthNetModule class.
    
    Returns:
        DepthNetModule: The initialized HybridDepth model with optional pre-trained weights.
    """
    model = DepthNetModule(**kwargs)
    if pretrained:
        pretrained_resource = "https://github.com/cake-lab/HybridDepth/releases/download/v2.0/NyuBest5.ckpt"
        checkpoint = torch.hub.load_state_dict_from_url(pretrained_resource, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
    return model
