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
"""Spriteworld sprite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from matplotlib import path as mpl_path
from matplotlib import transforms as mpl_transforms
import numpy as np
from spriteworld import constants

FACTOR_NAMES = (
    'x',  # x-position of sprite center-of-mass (float)
    'y',  # y-position of sprite center-of-mass (float)
    'shape',  # shape (string)
    'angle',  # angle in degrees (scalar)
    'scale',  # size of sprite (float)
    'c0',  # first color component (scalar)
    'c1',  # second color component (scalar)
    'c2',  # third color component (scalar)
    'x_vel',  # x-component of velocity (float)
    'y_vel',  # y-component of velocity (float)
)

# Just to catch infinite while-looping. Anything >1e4 should be plenty safe.
_MAX_TRIES = int(1e6)


class Sprite(object):
  """Sprite class.

  Sprites are simple shapes parameterized by a few factors (position, shape,
  angle, scale, color, velocity). They are the building blocks of Spriteworld,
  so every Spriteworld environment state is simple a collection of sprites.

  We assume that (x, y) are in mathematical coordinates, i.e. (0, 0) is at the
  lower-left of the frame.
  """

  def __init__(self,
               x=0.5,
               y=0.5,
               shape='square',
               angle=0,
               scale=0.1,
               c0=0,
               c1=0,
               c2=0,
               x_vel=0.0,
               y_vel=0.0,
               goal_x=None,
               goal_y=None,
               move_noise=0,
               is_barrier=False,
               barrier_stretch=1.,
               max_abs_vel=0.08):
    """Construct sprite.

    This class is agnostic to the color scheme, namely (c1, c2, c3) could be in
    RGB coordinates or HSV, HSL, etc. without this class knowing. The color
    scheme conversion for rendering must be done in the renderer.

    Args:
      x: Float in [0, 1]. x-position.
      y: Float in [0, 1]. y-position.
      shape: String. Shape of the sprite. Must be a key of constants.SHAPES.
      angle: Int. Angle in degrees.
      scale: Float in [0, 1]. Scale of the sprite, from a point to the area of
        the entire frame. This scales linearly with respect to sprite width,
        hence with power 1/2 with respect to sprite area.
      c0: Scalar. First coordinate of color.
      c1: Scalar. Second coordinate of color.
      c2: Scalar. Third coordinate of color.
      x_vel: Float. x-velocity.
      y_vel: Float. y-velocity.
      goal_x: Float in [0, 1]. target x-position.
      goal_y: Float in [0, 1]. target y-position.
    """
    self._position = np.array([x, y])
    self._goal_position = np.array([goal_x, goal_y])
    self._shape = shape
    self._angle = angle
    self._scale = scale
    self._color = (c0, c1, c2)
    self._velocity = np.array((x_vel, y_vel))
    self._max_abs_vel = max_abs_vel
    self._is_barrier = is_barrier
    if is_barrier:
      self._color = (255, 255, 255)
    self._barrier_stretch = barrier_stretch
    self._reset_centered_path()

    self._move_noise = move_noise
    self._goal = True
    if self._goal_position[0] is None:
      self._goal = False

    self.influenced_by_on_last_move = None


  def _reset_centered_path(self):
    path = mpl_path.Path(constants.SHAPES[self._shape])
    scale_rotate = (
        mpl_transforms.Affine2D().scale(self._scale*self._barrier_stretch, self._scale) +
        mpl_transforms.Affine2D().rotate_deg(self._angle))
    self._centered_path = scale_rotate.transform_path(path)

  def move(self, motion, keep_in_frame=False, barriers=[], prevent_intersect=-1, acted_on=False):
    """Move the sprite, optionally keeping its centerpoint within the frame."""
    old_position = self._position.copy()

    self._position += motion
    if self._move_noise:
      self._position += np.random.normal(loc=0.0, scale=self._move_noise, size=self._position.shape)

    if keep_in_frame:
      if np.any(self._position >= 1.0) or np.any(self._position <=0.):
        self.reverse_velocity(0.)
      self._position = np.clip(self._position, 0.0, 1.0)

    influenced_by_on_last_move = np.zeros((len(barriers) + 1,),
                                          dtype=np.float32)

    hit_something = False
    if barriers:
      for i, sprite in enumerate(barriers):
        if sprite is self:
          influenced_by_on_last_move[i] = 1
        elif not hit_something and prevent_intersect > 0 and np.linalg.norm(self._position - sprite.position) < prevent_intersect:
          if not np.all(self._position == old_position) or np.any(self.velocity):
            influenced_by_on_last_move[i] = 1
          self._position = old_position
          self.reverse_velocity(0.)
          hit_something = True
        elif not hit_something and sprite.contains_point(self._position):
          if not np.all(self._position == old_position) or np.any(self.velocity):
            influenced_by_on_last_move[i] = 1
          self._position = old_position
          self.reverse_velocity(0.)
          hit_something = True
    
    influenced_by_on_last_move[-1] = float(acted_on)
    self.influenced_by_on_last_move = influenced_by_on_last_move

  def update_position(self, keep_in_frame=False, barriers=[], prevent_intersect=-1, acted_on=False):
    """Update position based on velocity."""
    self.move(self.velocity, keep_in_frame=keep_in_frame, barriers=barriers, prevent_intersect=prevent_intersect, acted_on=acted_on)

  def reverse_velocity(self, noise):
    self._velocity *= -1 # bounce the sprite
    self._velocity += np.random.normal(loc=[0., 0.], scale=noise)
    self._velocity = np.clip(self._velocity, -self._max_abs_vel, self._max_abs_vel)

  def contains_point(self, point):
    """Check if the point is contained in the Sprite."""
    return self._centered_path.contains_point(point - self.position)

  def distance_to_goal(self):
    """returns distance to goal"""
    if self._goal:
      return np.linalg.norm(self._position - self._goal_position)
    return 0.

  def sample_contained_position(self):
    """Sample random position uniformly within sprite."""
    low = np.min(self._centered_path.vertices, axis=0)
    high = np.max(self._centered_path.vertices, axis=0)
    for _ in range(_MAX_TRIES):
      sample = self._position + np.random.uniform(low, high)
      if self.contains_point(sample):
        return sample
    raise ValueError('max_tries exceeded. There is almost surely an error in '
                     'the SpriteWorld library code.')

  @property
  def vertices(self):
    """Numpy array of vertices of the shape."""
    transform = mpl_transforms.Affine2D().translate(*self._position)
    path = transform.transform_path(self._centered_path)
    return path.vertices

  @property
  def goal_vertices(self):
    transform = mpl_transforms.Affine2D().translate(*self._goal_position)
    path = transform.transform_path(self._centered_path)
    return path.vertices

  @property
  def out_of_frame(self):
    return not (np.all(self._position >= [0., 0.]) and
                np.all(self._position <= [1., 1.]))

  @property
  def x(self):
    return self._position[0]

  @property
  def y(self):
    return self._position[1]

  @property
  def shape(self):
    return self._shape

  @shape.setter
  def shape(self, s):
    self._shape = s
    self._reset_centered_path()

  @property
  def angle(self):
    return self._angle

  @property
  def is_barrier(self):
    return self._is_barrier

  @angle.setter
  def angle(self, a):
    rotate = mpl_transforms.Affine2D().rotate_deg(a - self._angle)
    self._centered_path = rotate.transform_path(self._centered_path)
    self._angle = a

  @property
  def scale(self):
    return self._scale

  @scale.setter
  def scale(self, s):
    rescale = mpl_transforms.Affine2D().scale(s - self._scale)
    self._centered_path = rescale.transform_path(self._centered_path)
    self._scale = s

  @property
  def c0(self):
    return self._color[0]

  @property
  def c1(self):
    return self._color[1]

  @property
  def c2(self):
    return self._color[2]

  @property
  def x_vel(self):
    return self._velocity[0]

  @property
  def y_vel(self):
    return self._velocity[1]

  @property
  def color(self):
    return self._color

  @property
  def position(self):
    return self._position

  @property
  def goal_position(self):
    if self._goal:
      return self._goal_position
    else:
      return self._position

  @property
  def velocity(self):
    return self._velocity

  def set_color(self, color):
    self._color = color # e.g., (255, 255, 255)
    return self

  @property
  def factors(self):
    factors = collections.OrderedDict()
    for factor_name in FACTOR_NAMES:
      factors[factor_name] = getattr(self, factor_name)
    return factors
