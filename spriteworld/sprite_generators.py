# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3
"""Generators for producing lists of sprites based on factor distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
from spriteworld import sprite

MAX_TRIES = 1000

def resample_if_in_barrier(sprite_generator):
  """resamples from the sprite_generator if it generates an initial position where
  any of the non-barrier sprites are inside a barrier"""

  def _generate():
    for _ in range(MAX_TRIES):
      sprites = sprite_generator()
      barriers = []
      nonbarriers = []
      for s in sprites:
        if s.is_barrier:
          barriers.append(s)
        else:
          nonbarriers.append(s)
      
      good_generation = True
      for s in nonbarriers:
        for barrier in barriers:
          if barrier.contains_point(s.position):
            good_generation = False
            break
          elif s._goal and barrier.contains_point(s.goal_position):
            good_generation = False
            break
      
      if good_generation:
        break

    return sprites

  return _generate


def generate_sprites(factor_dist, num_sprites=1):
  """Create callable that samples sprites from a factor distribution.

  Args:
    factor_dist: The factor distribution from which to sample. Should be an
      instance of factor_distributions.AbstractDistribution.
    num_sprites: Int or callable returning int. Number of sprites to generate
      per call.

  Returns:
    _generate: Callable that returns a list of Sprites.
  """

  def _generate():
    n = num_sprites() if callable(num_sprites) else num_sprites
    sprites = [sprite.Sprite(**factor_dist.sample()) for _ in range(n)]
    return sprites

  return _generate


def generate_nonintersecting_sprites(factor_dist, num_sprites=1, epsilon=0.1):
  """Create callable that samples sprites from a factor distribution.

  Args:
    factor_dist: The factor distribution from which to sample. Should be an
      instance of factor_distributions.AbstractDistribution.
    num_sprites: Int or callable returning int. Number of sprites to generate
      per call.

  Returns:
    _generate: Callable that returns a list of Sprites.
  """

  def _generate():
    n = num_sprites() if callable(num_sprites) else num_sprites
    sprites = []
    while len(sprites) < num_sprites:
      s = sprite.Sprite(**factor_dist.sample())
      accept=True
      for other_s in sprites:
        if np.linalg.norm(s.position - other_s.position) < epsilon:
          accept=False
          break
      if accept:
        sprites.append(s)
    sprites = [sprite.Sprite(**factor_dist.sample()) for _ in range(n)]
    return sprites

  return _generate


def chain_generators(*sprite_generators):
  """Chain generators by concatenating output sprite sequences.

  Essentially an 'AND' operation over sprite generators. This is useful when one
  wants to control the number of samples from the modes of a multimodal sprite
  distribution.

  Note that factor_distributions.Mixture provides weighted mixture
  distributions, so chain_generators() is typically only used when one wants to
  forces the different modes to each have a non-zero number of sprites.

  Args:
    *sprite_generators: Callable sprite generators.

  Returns:
    _generate: Callable returning a list of sprites.
  """

  def _generate():
    return list(
        itertools.chain(*[generator() for generator in sprite_generators]))

  return _generate


def sample_generator(sprite_generators, p=None):
  """Sample one element from a set of sprite generators.

  Essential an 'OR' operation over sprite generators. This returns a callable
  that samples a generator from sprite_generators and calls it.

  Note that if sprite_generators each return 1 sprite, this functionality can be
  achieved with factor_distributions.Mixture, so sample_generator is typically
  used when sprite_generators each return multiple sprites. Effectively it
  allows dependant sampling from a multimodal factor distribution.

  Args:
    sprite_generators: Iterable of callable sprite generators.
    p: Probabilities associated with each generator. If None, assumes uniform
      distribution.

  Returns:
    _generate: Callable sprite generator.
  """

  def _generate():
    sample_index = np.random.choice(len(sprite_generators), p=p)
    sampled_generator = sprite_generators[sample_index]
    return sampled_generator()

  return _generate


def shuffle(sprite_generator):
  """Randomize the order of sprites sample from sprite_generator.

  This is useful because sprites are z-layered with occlusion according to their
  order, so is sprite_generator is the output of chain_generators(), then
  sprites from some component distributions will always be behind sprites from
  others.

  An alternate design would be to let the environment handle sprite ordering,
  but this design is preferable because the order can be controlled more finely.
  For example, this allows the user to specify one sprite (e.g. the agent's
  body) to always be in the foreground while all the others are randomly
  ordered.

  Args:
    sprite_generator: Callable return a list of sprites.

  Returns:
    _generate: Callable sprite generator.
  """

  def _generate():
    sprites = sprite_generator()
    order = np.arange(len(sprites))
    np.random.shuffle(order)
    return [sprites[i] for i in order]

  return _generate

def sort_by_color(sprite_generator):
  """Sort sprites from generator by color (in order c1, c2, c3)

  Args:
    sprite_generator: Callable return a list of sprites.

  Returns:
    _generate: Callable sprite generator.
  """

  def _generate():
    sprites = sprite_generator()
    
    return sorted(sprites, key=lambda x: (x.c0, x.c1, x.c2))

  return _generate
