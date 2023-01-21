import torch
from torch.nn.utils import prune

class ThresholdPruning(prune.BasePruningMethod):
	PRUNING_TYPE = "unstructured"

	def __init__(self, threshold):
		self.threshold = threshold

	def compute_mask(self, tensor, default_mask):
		return torch.abs(tensor) > self.threshold
