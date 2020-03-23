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
"""Action spaces for Spriteworld.

This file contains action space classes compatible with Spriteworld.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_env import specs
import numpy as np

class SelectBounce(object):
  """Select-Bounce action space.

  This action space takes a continuous position of length 2, with each component
  in [0, 1]. This can be intuited as representing a single click (x, y) coordinate. 

  If there is a sprite where the screen is clicked, that sprite's velocities are 
  reversed.
  """

  def __init__(self, noise_scale=0.01, prevent_intersect=0.1):
    """Constructor.

    Args:
      scale: Multiplier by which the motion is scaled down. Should be in [0.0, 1.0].
      noise_scale: Optional stddev of the noise. If scalar, applied to all
        action space components. If vector, must have same shape as action.
    """
    self._noise_scale = noise_scale
    self._action_spec = specs.BoundedArray(shape=(2,), dtype=np.float32, minimum=0.0, maximum=1.0)
    self._prevent_intersect = prevent_intersect

  def get_sprite_from_position(self, position, sprites):
    for sprite in sprites:
      if sprite.contains_point(position):
        return sprite
    return None

  def step(self, action, sprites, *unused_args, **unused_kwargs):
    """Take an action and move the sprites.

    Args:
      action: Numpy array of shape (2,) in [0, 1]. 
      sprites: Iterable of sprite.Sprite() instances.

    Returns:
      Scalar cost of taking this action.
    """
    if self._prevent_intersect > 0:
      barriers = sprites
    else:
      barriers = []
    clicked_sprite = self.get_sprite_from_position(action, sprites)
    if clicked_sprite is not None:
      clicked_sprite.reverse_velocity(self._noise_scale)
      
    for sprite in sprites:
      sprite.update_position(keep_in_frame=True, barriers=barriers, prevent_intersect=self._prevent_intersect, acted_on=(sprite is clicked_sprite))
    return 0.

  def sample(self):
    """Sample an action uniformly randomly."""
    return np.random.uniform(0., 1., size=(2,))

  def action_spec(self):
    return self._action_spec



class SelectMove(object):
  """Select-Move action space.

  This action space takes in a continuous vector of length 4 with each component
  in [0, 1]. This can be intuited as representing two consecutive clicks:
    [first_x, first_y, second_x, second_y].

  These two clicks are then processed to generate a position and a motion:
    * Position = [first_x, first_y]
    * Motion = scale * [second_x - 0.5, second_y - 0.5]

  If the Position, viewed as a point in the arena, lies inside of a sprite, that
  sprite will be moved by Motion, which is a scaled version of the second click
  relative to the center of the arena. If the Position does not lie inside of a
  sprite then no sprite will move. So to move a sprite you have to click on it
  and click on the direction you want it to move, like a touch screen.

  There is an optional control cost proportional to the norm of the motion.
  """

  def __init__(self, scale=1.0, motion_cost=0.0, noise_scale=None, move_all=False, prevent_intersect=-1):
    """Constructor.

    Args:
      scale: Multiplier by which the motion is scaled down. Should be in [0.0,
        1.0].
      motion_cost: Factor by which motion incurs cost.
      noise_scale: Optional stddev of the noise. If scalar, applied to all
        action space components. If vector, must have same shape as action.
    """
    self._scale = scale
    self._motion_cost = motion_cost
    self._noise_scale = noise_scale
    self._action_spec = specs.BoundedArray(
        shape=(4,), dtype=np.float32, minimum=0.0, maximum=1.0)
    self._move_all = move_all
    self._prevent_intersect = prevent_intersect

  def get_motion(self, action):
    delta_pos = (action[2:] - 0.5) * self._scale
    return delta_pos

  def apply_noise_to_action(self, action):
    if self._noise_scale:
      noise = np.random.normal(
          loc=0.0, scale=self._noise_scale, size=action.shape)
      return action + noise
    return action

  def get_sprite_from_position(self, position, sprites):
    for sprite in sprites[::-1]:
      if sprite.contains_point(position):
        return sprite
    return None

  def step(self, action, sprites, keep_in_frame):
    """Take an action and move the sprites.

    Args:
      action: Numpy array of shape (4,) in [0, 1]. First two components are the
        position selection, second two are the motion selection.
      sprites: Iterable of sprite.Sprite() instances. If a sprite is moved by
        the action, its position is updated.
      keep_in_frame: Bool. Whether to force sprites to stay in the frame by
        clipping their centers of mass to be in [0, 1].

    Returns:
      Scalar cost of taking this action.
    """
    if self._prevent_intersect > 0:
      barriers = sprites
    else:
      barriers = []
    
    noised_action = self.apply_noise_to_action(action)
    position = noised_action[:2]
    motion = self.get_motion(noised_action)
    clicked_sprite = self.get_sprite_from_position(position, sprites)
    if clicked_sprite is not None:
      clicked_sprite.move(motion, keep_in_frame=keep_in_frame, barriers=barriers, prevent_intersect=self._prevent_intersect)
    if self._move_all:
      for sprite in sprites:
        sprite.move(0., keep_in_frame, barriers=barriers, prevent_intersect=self._prevent_intersect)
    return -self._motion_cost * np.linalg.norm(motion)

  def sample(self):
    """Sample an action uniformly randomly."""
    return np.random.uniform(0., 1., size=(4,))

  def action_spec(self):
    return self._action_spec


class DragAndDrop(SelectMove):
  """Drag-And-Drop action space.

  This action space takes in a continuous vector of length 4 with each component
  in [0, 1]. This can be intuited as representing two consecutive clicks:
    [first_x, first_y, second_x, second_y].

  These two clicks are then processed to generate a position and a motion:
  * Position = [first_x, first_y]
  * Motion = scale * [second_x - first_x, second_y - first_y]

  * Target = [second_x, second_y]

  Similarly to SelectMove, a sprite will only move if Position lies in it. The
  only difference is here the Motion is relative to the Position, instead of
  relative to the center of the screen. So the second click effectively
  specifies a target location towards which the sprite moves.
  """

  def get_motion(self, action):
    pos = action[:2]
    target = action[2:]
    delta_pos = (target - pos) * self._scale
    return delta_pos


class Embodied(object):
  """Embodied-Grid action space.

  This action space treats sprites[-1] (the foreground sprite) as the agent's
  body.

  The action space has two components. The first is a binary `Carry/Don't Carry`
  component which allows the agent to carry the sprite immediately beneath it as
  it moves, if there is such a sprite. The second controls the agent's motion
  and consists of `Up/Down/Left/Right` options.
  """

  def __init__(self, step_size=0.05, motion_cost=0.):
    """Constructor.

    Args:
      step_size: Fraction of the arena width the sprite moves for each step.
      motion_cost: Each step incurs cost motion_cost * step_size.
    """
    self._step_size = step_size
    self._motion_cost = motion_cost
    self._action_spec = [
        specs.DiscreteArray(num_values=2, dtype=np.int64),
        specs.DiscreteArray(num_values=4, dtype=np.int64),
    ]
    self.action_to_motion = {
        0: np.array([0, self._step_size]),   # Up
        1: np.array([-self._step_size, 0]),  # Left
        2: np.array([0, -self._step_size]),  # Down
        3: np.array([self._step_size, 0]),   # Right
    }

  def get_body_sprite(self, sprites):
    """Return the sprite representing the agent's body."""
    return sprites[-1]

  def get_non_body_sprites(self, sprites):
    """Return all sprites except that representing the agent's body."""
    return sprites[:-1]

  def get_carried_sprite(self, sprites):
    body_position = self.get_body_sprite(sprites).position
    for sprite in self.get_non_body_sprites(sprites)[::-1]:
      if sprite.contains_point(body_position):
        return sprite
    return None

  def step(self, action, sprites, keep_in_frame):
    """Take an action and move the sprites.

    Args:
      action: Iterable of length 2. First component must be in [0, 1] and second
        component must be in [0, 1, 2, 3].
      sprites: Iterable of sprite.Sprite() instances. sprites[-1] is the agent's
        body.
      keep_in_frame: Bool. Whether to force sprites to stay in the frame by
        clipping their centers of mass to be in [0, 1].

    Returns:
      Scalar cost of taking this action.
    """

    carry = action[0]
    motion = self.action_to_motion[action[1]]

    # Move carried sprite if necessary
    if carry:
      carried_sprite = self.get_carried_sprite(sprites)
      if carried_sprite is not None:
        carried_sprite.move(motion, keep_in_frame=keep_in_frame)

    # Move agent body
    self.get_body_sprite(sprites).move(motion, keep_in_frame=keep_in_frame)

    return -self._motion_cost * self._step_size

  def sample(self):
    """Sample an action uniformly randomly."""
    return [np.random.randint(0, 2), np.random.randint(0, 4)]

  def action_spec(self):
    return self._action_spec


class Navigate(object):
  """Continuous navigation action.

  This action space treats sprites[-1] (the foreground sprite) as the agent's
  body.

  Each action is a vector in [-1, 1] x [-1, 1], which specifies the direction the
  agent desires to move in.
  """

  def __init__(self, step_size=0.05, action_noise_percent=0.1, motion_cost=1., slow_zones=[], slow_factor=0.2):
    """Constructor.

    Args:
      step_size: Fraction of the arena width the sprite moves for each step.
      motion_cost: Each step incurs cost motion_cost * step_size.
      slow_zones: list of ((lower_x, lower_y), (upper_x, upper_y)) that define areas through which
        things move slowly (20% of speed)
    """
    self._step_size = step_size
    self._motion_cost = motion_cost
    self._action_noise = action_noise_percent

    self._action_spec = specs.BoundedArray((2,), np.float32, -1., 1.)

    self._slow_zones = list(map(np.array, slow_zones))
    self._slow_factor = slow_factor
    
  def get_body_sprite(self, sprites):
    """Return the sprite representing the agent's body."""
    return sprites[-1]

  def get_non_body_sprites(self, sprites):
    """Return all sprites except that representing the agent's body."""
    return sprites[:-1]

  def get_intersecting_sprites(self, sprites):
    body_position = self.get_body_sprite(sprites).position
    res = []
    for sprite in self.get_non_body_sprites(sprites):
      if sprite.contains_point(body_position):
        res.append(sprite)
    return res

  def split_sprites(self, sprites):
    """splits sprites into barrier and non-barrier sprites"""
    barriers = []
    nonbarriers = []
    for sprite in sprites:
      if sprite.is_barrier:
        barriers.append(sprite)
      else:
        nonbarriers.append(sprite)
    return barriers, nonbarriers

  def in_slow_zone(self, sprite):
    res = False
    for (lower, upper) in self._slow_zones:
      if np.all(sprite.position > lower) and np.all(sprite.position < upper):
        res=True
        break
    return res

  def step(self, action, sprites, keep_in_frame=True):
    """Take an action and move the sprites.

    Args:
      action: 2d movement
      sprites: Iterable of sprite.Sprite() instances. sprites[-1] is the agent's
        body.
      keep_in_frame: Bool. Whether to force sprites to stay in the frame by
        clipping their centers of mass to be in [0, 1].

    Returns:
      Scalar cost of taking this action.
    """
    barriers, nonbarriers = self.split_sprites(sprites)

    # Add action noise
    action *= (1 + np.random.uniform(low=-self._action_noise, high=self._action_noise, size=2))
    agent = self.get_body_sprite(sprites)
    if self.in_slow_zone(agent):
      action *= self._slow_factor

    # If move is slow enough, move intersecting sprites
    if np.all(np.abs(action) < 0.8):
      for sprite in self.get_intersecting_sprites(nonbarriers):
        sprite.move(action * self._step_size, keep_in_frame=keep_in_frame, barriers=barriers)

    # Move agent body, rejecting bad moves
    agent.move(action * self._step_size, keep_in_frame=keep_in_frame, barriers=barriers)

    return -self._motion_cost

  def sample(self):
    """Sample an action uniformly randomly."""
    return np.random.rand(2) * 2. - 1.

  def action_spec(self):
    return self._action_spec
