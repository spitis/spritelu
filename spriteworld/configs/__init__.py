"""__init__.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from spriteworld import environment, sprite, tasks, action_spaces, renderers as spriteworld_renderers
import os, copy, numpy as np

def action_space():
  return action_spaces.Navigate()

def task():
  return tasks.SparseGoalPlacement()

def image_renderers():
  return {
    'observation': spriteworld_renderers.PILRenderer((64,64), anti_aliasing=5),
    'achieved_goal': spriteworld_renderers.AchievedPILGoalRenderer((64,64), anti_aliasing=5),
    'desired_goal': spriteworld_renderers.PILGoalRenderer((64, 64), anti_aliasing=5)
  }

def disentangled_renderers():
  return {
    'observation': spriteworld_renderers.VectorizedPositions(),
    'achieved_goal': spriteworld_renderers.AchievedVectorizedPositions(),
    'desired_goal': spriteworld_renderers.VectorizedGoalPositions()
  }

def random_vector_renderers():

  random_mtx = (np.random.rand(100, 100) - 0.5)*2.
  fn=lambda a: np.dot(random_mtx[:len(a),:len(a)], a)

  return {
    'observation': spriteworld_renderers.VectorizedPositions(),
    'achieved_goal': spriteworld_renderers.AchievedFunctionOfVectorizedPositions(fn=fn),
    'desired_goal': spriteworld_renderers.FunctionOfVectorizedGoalPositions(fn=fn)
  }

s1 = sprite.Sprite(0.25, 0.25, 'triangle', c0=140,c1=220, c2=80, goal_x=0.1, goal_y=0.1)
s2 = sprite.Sprite(0.9, 0.5, 'square', c0=200,c2=255)
s3 = sprite.Sprite(0.25, 0.75, 'circle', c2=255, c1=210, goal_x=0.1, goal_y=0.9)
s4 = sprite.Sprite(0.50, 0.75, 'star_5', c0=255, c1=80)

b1 = sprite.Sprite(0.4, 0.4, is_barrier=True)
b2 = sprite.Sprite(0.3, 0.4, is_barrier=True)
b3 = sprite.Sprite(0.2, 0.4, is_barrier=True)

init_sprites = lambda: copy.deepcopy((b1, b2, b3, s1, s2, s3, s4))
simple_sprites = lambda: copy.deepcopy((s3, s4))

test_config = {
    'task': tasks.SparseGoalPlacement(),
    'action_space': action_spaces.Navigate(),
    'renderers': image_renderers(),
    'init_sprites': init_sprites,
    'max_episode_length': 1000,
    'metadata': {
        'name': os.path.basename(__file__)
    }
}
simple_config = {
    'task': tasks.SparseGoalPlacement(),
    'action_space': action_spaces.Navigate(),
    'renderers': image_renderers(),
    'init_sprites': simple_sprites,
    'max_episode_length': 1000,
    'metadata': {
        'name': os.path.basename(__file__)
    }
}

def renderers():
  return {
      'image':
          spriteworld_renderers.PILRenderer(
              image_size=(64, 64),
              anti_aliasing=5,
              color_to_rgb=spriteworld_renderers.color_maps.hsv_to_rgb,
          )
  }

