"""Goal renderers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env import specs
import numpy as np
from spriteworld.renderers import abstract_renderer, VectorizedPositions, FunctionOfVectorizedPositions 
from spriteworld.renderers import PILRenderer
from PIL import Image

class AchievedObservationRenderer(abstract_renderer.AbstractRenderer):
  """Renders a copy of the observation."""

  def __init__(self, render_nongoals=False):
    """Constructor."""
    self._observation_spec = specs.Array(shape=(), dtype=np.object)

  def render(self, sprites=(), global_state=None):
    """Copies the observation from the global state
    """
    obs = global_state[1]['observation']
    if self._observation_spec is None:
      self._observation_spec = specs.Array(shape=obs.shape, dtype=obs.dtype)
    return obs

  def observation_spec(self):
    return self._observation_spec

class PILGoalRenderer(PILRenderer):
  """Renders the goal"""

  def render(self, sprites=(), global_state=None):
    self._canvas.paste(self._canvas_bg)
    for obj in sprites:
      if obj._goal or self._render_nongoals:
        vertices = self._canvas_size * obj.goal_vertices
        color = self._color_to_rgb(obj.color)
        self._draw.polygon([tuple(v) for v in vertices], fill=color)
    image = self._canvas.resize(self._image_size, resample=Image.ANTIALIAS)

    # PIL uses a coordinate system with the origin (0, 0) at the upper-left, but
    # our environment uses an origin at the bottom-left (i.e. mathematical
    # convention). Hence we need to flip the render vertically to correct for
    # that.
    image = np.flipud(np.array(image))
    return image

class AchievedPILGoalRenderer(PILRenderer):
  def render(self, sprites=(), global_state=None):
    goal_sprites = [sprite for sprite in sprites if sprite._goal]
    return super().render(goal_sprites, global_state)

class VectorizedGoalPositions(abstract_renderer.AbstractRenderer):
  """Aggregates positions of the sprites into an array."""

  def __init__(self, render_nongoals=False):
    """Constructor.
    """
    self._num_sprites = None
    self._render_nongoals = render_nongoals

  def render(self, sprites=(), global_state=None):
    """Renders a list of sprites into an array where every two components is an xy position.

    Args:
      sprites: a list of sprites
      global_state: Unused global state.

    Returns:
      An array of sprite goal_positions
    """
    # Set number of sprites so that observation_spec is callable
    self._num_sprites = len(sprites)

    res =  np.array([sprite.goal_position for sprite in sprites if 
      (sprite._goal or self._render_nongoals)]).flatten()
    self._render_shape = res.shape

    return res

  def observation_spec(self):
    return specs.Array(shape=self._render_shape, dtype=np.float32)


class AchievedVectorizedPositions(VectorizedPositions):
  def render(self, sprites=(), global_state=None):
    goal_sprites = [sprite for sprite in sprites if sprite._goal]
    return super().render(goal_sprites, global_state)

class FunctionOfVectorizedGoalPositions(abstract_renderer.AbstractRenderer):
  """Aggregates positions of the sprites into an array."""

  def __init__(self, fn, render_nongoals=False):
    """Constructor.
    """
    self._fn = fn
    self._observation_spec = specs.Array(shape=(), dtype=np.object)
    self._render_nongoals = render_nongoals

  def render(self, sprites=(), global_state=None):
    """Renders a list of sprites into an array where every two components is an xy position.

    Args:
      sprites: a list of sprites
      global_state: Unused global state.

    Returns:
      An array of sprite positions
    """
    # Set number of sprites so that observation_spec is callable
    vec_pos = np.array([sprite.goal_position for sprite in sprites if 
      (sprite._goal or self._render_nongoals)]).flatten()
    obs = self._fn(vec_pos)
    
    if self._observation_spec is None:
      self._observation_spec = specs.Array(shape=obs.shape, dtype=obs.dtype)
    return obs

  def observation_spec(self):
    return self._observation_spec


class AchievedFunctionOfVectorizedPositions(FunctionOfVectorizedPositions):
  def render(self, sprites=(), global_state=None):
    goal_sprites = [sprite for sprite in sprites if sprite._goal]
    return super().render(goal_sprites, global_state)