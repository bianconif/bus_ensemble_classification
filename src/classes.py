from dataclasses import dataclass

@dataclass
class DescriptorWrapper:
    """Wrapper around an image descriptor.
    
    Attributes
    ----------
    descriptor: object
        The descriptor object -- i.e., the object that computes the 
        features
    mode: str
        Whether the features should be computed from the image or the roi.
        Can be either 'image' or 'roi'
    """
    descriptor: object
    mode: str