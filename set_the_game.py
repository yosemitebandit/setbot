"""Defines the rules of set."""


def is_set(card_one, card_two, card_three):
  """Checks if three cards constitute a set.

  Args:
    each card is a dict with four required keys.
  """
  cards = (card_one, card_two, card_three)
  # Capture all unique attributes in the set of three.
  numbers = set([c['number'] for c in cards])
  colors = set([c['color'] for c in cards])
  fills = set([c['fill'] for c in cards])
  shapes = set([c['shape'] for c in cards])
  # Compare.
  if (len(numbers) == 2 or
      len(colors) == 2 or
      len(fills) == 2 or
      len(shapes) == 2):
    return False
  else:
    return True
