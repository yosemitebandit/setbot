"""Tests the SetGame class."""

import unittest

import set_the_game


class SetGameTest(unittest.TestCase):

  def test_is_set_one(self):
    """A set can be three cards with just one variation among them."""
    card_one = {
      'number': '1',
      'color': 'red',
      'fill': 'empty',
      'shape': 'diamond',
    }
    card_two = dict(card_one)
    card_two['number'] = '2'
    card_three = dict(card_one)
    card_three['number'] = '3'
    self.assertTrue(set_the_game.is_set(card_one, card_two, card_three))

  def test_is_set_two(self):
    """A set is also three cards that are totally different."""
    card_one = {
      'number': '1',
      'color': 'red',
      'fill': 'empty',
      'shape': 'diamond',
    }
    card_two = {
      'number': '2',
      'color': 'green',
      'fill': 'solid',
      'shape': 'oval',
    }
    card_three = {
      'number': '3',
      'color': 'purple',
      'fill': 'striped',
      'shape': 'bean',
    }
    self.assertTrue(set_the_game.is_set(card_one, card_two, card_three))

  def test_is_set_three(self):
    """But two variations among three cards is not a set."""
    card_one = {
      'number': '1',
      'color': 'red',
      'fill': 'empty',
      'shape': 'diamond',
    }
    card_two = dict(card_one)
    card_two['number'] = '2'
    card_three = dict(card_one)
    card_three['number'] = '3'
    card_three['color'] = 'purple'
    self.assertFalse(set_the_game.is_set(card_one, card_two, card_three))
