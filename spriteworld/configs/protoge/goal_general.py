from spriteworld import environment, sprite, tasks, action_spaces, sprite_generators
from spriteworld import renderers as spriteworld_renderers, factor_distributions as distribs
import os, copy, numpy as np

TERMINATE_DISTANCE = 0.075

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

def get_config(num_goal_objects=1, num_barriers=0, num_distractors=0, agent_has_goal=False, scale=0.1):
  """Generate environment config.

  Args:
    num_goal_objects: number of objects that need to be placed
    num_barriers: number of random barriers to spawn
    num_distractors: number of distractor objects (do not need to be placed)
    agent_has_goal: Whether or not the agent also needs to be placed at a target location

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
    distribs.Continuous('c2', 25, 230),
    distribs.Discrete('scale', [scale])
  ])

  goal_loc = distribs.Product([
    distribs.Continuous('goal_x', 0.1, 0.9),
    distribs.Continuous('goal_y', 0.1, 0.9),
  ])

  ## OBJECTS
  goal_factors = distribs.Product([
    shared_factors,
    distribs.Discrete('shape', ['square', 'triangle', 'circle']),
    goal_loc
  ])
  goal_sprite_gen = sprite_generators.generate_sprites(goal_factors, num_sprites=num_goal_objects)
  gen_list.append(goal_sprite_gen )

  ## BARRIERS
  barrier_factors = distribs.Product([
    shared_factors,
    distribs.Continuous('barrier_stretch', 2., 10.),
    distribs.Continuous('angle', 0., 360),
    distribs.Discrete('is_barrier', [True])
  ])
  barrier_sprite_gen = sprite_generators.generate_sprites(barrier_factors, num_sprites=num_barriers)
  gen_list.append(barrier_sprite_gen)

  ## DISTRACTORS
  distractor_factors = distribs.Product([
    shared_factors,
    distribs.Discrete('shape', ['square', 'triangle', 'circle']),
  ])
  distractor_sprite_gen = sprite_generators.generate_sprites(distractor_factors, num_sprites=num_distractors)
  gen_list.append(distractor_sprite_gen)

  ## AGENT
  if agent_has_goal:
    agent_factors = distribs.Product([
      shared_factors,
      distribs.Discrete('shape', ['star_5']),
      goal_loc,
    ])
  else:
    agent_factors = distribs.Product([
      shared_factors,
      distribs.Discrete('shape', ['star_5']),
    ])
  agent_sprite_gen = sprite_generators.generate_sprites(agent_factors, num_sprites=1)

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
    }
  }

  return config
