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
"""Goal-Finding tasks testing for generalization to more distractors.

In this task there are two target sprites per episode of orange-green-ish color.
Those sprites must be brought to the goal location, which is the center of the
arena. There are also distractor sprites, which are blue-purple-ish color and do
not contribute to the reward. In train mode there is 1 distractor, while in test
mode there are two.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from spriteworld import environment
from spriteworld import sprite
from spriteworld import tasks
from spriteworld import action_spaces
from spriteworld import renderers as spriteworld_renderers
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators
from spriteworld.configs.cobra import common

import os, copy, numpy as np

TERMINATE_DISTANCE = 0.075

MODES_NUM_DISTRACTORS = {
    'train': 1,
    'test': 2,
}
MODES_NUM_GOALS = {
    'train':1,
    'test':1
}
MODES_NUM_BARRIERS = {
    'train':1,
    'test':1
}

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

def get_config(mode='train', level=0):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.
    level: int in [0,5] to vary the difficulty

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """
  gen_list = []

  shared_factors = distribs.Product([
    distribs.Continuous('x', 0.1, 0.9),
    distribs.Continuous('y', 0.1, 0.9),
    distribs.Continuous('c0', 25, 230),
    distribs.Continuous('c1', 25, 230),
    distribs.Continuous('c2', 25, 230)
  ])

  goal_loc = distribs.Product([
    distribs.Continuous('goal_x', 0.1, 0.9),
    distribs.Continuous('goal_y', 0.1, 0.9),
  ])

  if level > 0:
    # Agent is not part of goal
    agent_factors = distribs.Product([
      shared_factors,
      distribs.Discrete('shape', ['star_5']),
    ])

    # Have goal objects aside from the agent
    goal_factors = distribs.Product([
      shared_factors,
      goal_loc,
      distribs.Discrete('shape', ['square', 'triangle', 'circle']),
    ])
    goal_sprite_gen = sprite_generators.generate_sprites(
      goal_factors, num_sprites=MODES_NUM_GOALS[mode])
    gen_list.append(goal_sprite_gen )

    if level > 1:
      # Have barriers
      barrier_factors = distribs.Product([
        shared_factors,
        distribs.Continuous('barrier_stretch', 2., 10.),
        distribs.Continuous('angle', 0., 90),
        distribs.Discrete('is_barrier', [True])
      ])
      barrier_sprite_gen = sprite_generators.generate_sprites(
        barrier_factors, num_sprites=MODES_NUM_BARRIERS[mode])
      gen_list.append(barrier_sprite_gen)

    if level > 2:
      # Have distractor objects
      distractor_factors = distribs.Product([
        shared_factors,
        distribs.Discrete('shape', ['square', 'triangle', 'circle']),
      ])
      distractor_sprite_gen = sprite_generators.generate_sprites(
        distractor_factors, num_sprites=MODES_NUM_DISTRACTORS[mode])
      gen_list.append(distractor_sprite_gen)

  else:
    # Only the agent is the goal
    agent_factors = distribs.Product([
      shared_factors,
      distribs.Discrete('shape', ['star_5']),
      goal_loc
    ])


  agent_sprite_gen = sprite_generators.generate_sprites(
      agent_factors, num_sprites=1)


  # TODO: Sample the number of sprites as well
  sprite_gen = sprite_generators.chain_generators(*gen_list)

  # Randomize sprite ordering to eliminate any task information from occlusions
  sprite_gen = sprite_generators.shuffle(sprite_gen)
  
  # Add the agent in at the end
  sprite_gen = sprite_generators.resample_if_in_barrier(
    sprite_generators.chain_generators(sprite_gen, agent_sprite_gen))

  config = {
    'task': tasks.SparseGoalPlacement(epsilon=TERMINATE_DISTANCE),
    'action_space': action_spaces.Navigate(),
    'renderers': image_renderers(),
    'init_sprites': sprite_gen,
    'max_episode_length': 1000,
    'metadata': {
        'name': os.path.basename(__file__),
        'mode': mode

    }
  }

  return config
