"""Goal renderers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env import specs
import numpy as np
from spriteworld.renderers import abstract_renderer
from spriteworld.renderers import PILRenderer
from PIL import Image

class AchievedGoalRenderer(abstract_renderer.AbstractRenderer):
  """Renders a copy of the observation."""

  def __init__(self):
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