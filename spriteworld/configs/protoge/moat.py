import os
from spriteworld.environment import Environment
from spriteworld.gym_wrapper import GymWrapper
from spriteworld import tasks
from spriteworld import action_spaces
from spriteworld import renderers as spriteworld_renderers
from spriteworld import factor_distributions as distribs
from spriteworld import sprite_generators

import os, copy, numpy as np

TERMINATE_DISTANCE = 0.075


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


def get_config(mode = None, *unused_args):
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
      distribs.Continuous('x', 0.05, 0.35),
      distribs.Continuous('y', 0.05, 0.35),
      distribs.Continuous('c0', 25, 230),
      distribs.Continuous('c1', 25, 230),
      distribs.Continuous('c2', 25, 230)
  ])

  goal_loc = distribs.Product([
      distribs.Continuous('goal_x', 0.65, 0.95),
      distribs.Continuous('goal_y', 0.65, 0.95),
  ])

  agent_factors = distribs.Product([shared_factors, distribs.Discrete('shape', ['star_5']), goal_loc])

  agent_sprite_gen = sprite_generators.generate_sprites(agent_factors, num_sprites=1)

  action_space = action_spaces.Navigate(slow_zones=[((0.4, 0.4), (1., 0.6)), ((0.4, 0.4), (0.6, 1.))])

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
      'max_episode_length': 50,
      'metadata': {
          'name': os.path.basename(__file__)
      }
  }

  return config

def make_moat_env(config=None, seed=None):
  if config is None:
    config = get_config(mode='disentangled')
  gym_env = GymWrapper(Environment(**config, seed=seed))
  gym_env.compute_reward = lambda ag, g, info: -(np.linalg.norm(ag - g, axis=-1) > TERMINATE_DISTANCE).astype(np.float32)
  return gym_env