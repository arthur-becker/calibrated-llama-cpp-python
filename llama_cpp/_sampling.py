import numpy as np

def softmax(logits: np.array) -> np.array: 
    """
    Calculate the softmax of the logits

    Args:
    - logits: np.array of shape (num_classes,)

    Returns:
    - np.array of shape (num_classes,)
    """
    assert len(logits.shape) == 1, "logits must be a 1D array"

    exps = np.exp(logits - np.max(logits))
    result = exps / np.sum(exps)

    return result
