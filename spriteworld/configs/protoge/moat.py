import os
from spriteworld.environment import Environment
from spriteworld.gym_wrapper import GymWrapper
from spriteworld import tasks
from spriteworld import action_spaces
from spriteworld import renderers as spriteworld_renderers
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators

import os, copy, numpy as np

TERMINATE_DISTANCE = 0.05


def image_renderers():
  return {
      'observation': spriteworld_renderers.PILRenderer((64, 64), anti_aliasing=5),
      'achieved_goal': spriteworld_renderers.AchievedPILGoalRenderer((64, 64), anti_aliasing=5),
      'desired_goal': spriteworld_renderers.PILGoalRenderer((64, 64), anti_aliasing=5)
  }


def disentangled_renderers():
  return {
      'observation': spriteworld_renderers.VectorizedPositions(flatten=True),
      'achieved_goal': spriteworld_renderers.AchievedVectorizedPositions(flatten=True),
      'desired_goal': spriteworld_renderers.VectorizedGoalPositions()
  }


def random_vector_renderers():

  random_mtx = (np.random.rand(100, 100) - 0.5) * 2.
  fn = lambda a: np.dot(random_mtx[:len(a), :len(a)], a)

  return {
      'observation': spriteworld_renderers.VectorizedPositions(flatten=True),
      'achieved_goal': spriteworld_renderers.AchievedFunctionOfVectorizedPositions(fn=fn, flatten=True),
      'desired_goal': spriteworld_renderers.FunctionOfVectorizedGoalPositions(fn=fn)
  }


def get_config(mode = None, level = None, slow_factor = 0.2):
  """Generate environment config.

  Args:
    mode: 'train' or 'test'.
    level: int in [0,5] to vary the difficulty

  Returns:
    config: Dictionary defining task/environment configuration. Can be fed as
      kwargs to environment.Environment.
  """

  shared_factors = distribs.Product([
      distribs.Continuous('x', 0.05, 0.15),
      distribs.Continuous('y', 0.05, 0.15),
      distribs.Continuous('c0', 25, 230),
      distribs.Continuous('c1', 25, 230),
      distribs.Continuous('c2', 25, 230)
  ])

  goal_loc = distribs.Product([
      distribs.Continuous('goal_x', 0.85, 0.95),
      distribs.Continuous('goal_y', 0.85, 0.95),
  ])

  agent_factors = distribs.Product([shared_factors, distribs.Discrete('shape', ['star_5']), goal_loc])

  agent_sprite_gen = sprite_generators.generate_sprites(agent_factors, num_sprites=1)

  action_space = action_spaces.Navigate(slow_zones=[
    ((0.5, 0.5), (1., 0.8)), 
    ((0.5, 0.5), (0.8, 1.)),
    ((0., 0.9), (0.8, 1.)),
    ((0.9, 0.), (1., 0.8))
    ],
    slow_factor = slow_factor
    )

  if mode == 'disentangled':
    renderers = disentangled_renderers()
  elif mode == 'random_vectors':
    renderers = random_vector_renderers()
  else:
    renderers = image_renderers()

  config = {
      'task': tasks.SparseGoalPlacement(epsilon=TERMINATE_DISTANCE),
      'action_space': action_space,
      'renderers': renderers,
      'init_sprites': agent_sprite_gen,
      'max_episode_length': 100,
      'metadata': {
          'name': os.path.basename(__file__)
      },
      'reset_on_success': False
  }

  return config

def make_moat_env(config=None, seed=None, slow_factor=0.2):
  if config is None:
    config = get_config(mode='disentangled', slow_factor=slow_factor)
  gym_env = GymWrapper(Environment(**config, seed=seed))
  gym_env.compute_reward = lambda ag, g, info: -(np.linalg.norm(ag - g, axis=-1) > TERMINATE_DISTANCE).astype(np.float32)
  return gym_env