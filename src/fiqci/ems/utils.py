"""Utility functions"""


def probabilities_to_counts(probabilities, shots) -> list[dict]:
	"""Convert probabilities to counts"""
	try:
		probabilities[0]
	except KeyError:
		# If probabilities is not iterable, treat it as a single set of probabilities
		probabilities = [probabilities]

	counts_list = []
	for probs in probabilities:
		counts = {}
		for k, prob in probs.items():
			counts[k] = int(prob * shots)
		counts_list.append(counts)

	return counts_list
