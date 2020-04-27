# pylint: disable=g-bad-file-header
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
"""Start demo GUI for Spriteworld task configs.

To play a task, run this on the task config:
```bash
python run_demo.py --config=$path_to_task_config$
```

Be aware that this demo overrides the action space and renderer for ease of
playing, so those will be different from what are specified in the task config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
from absl import app
from absl import flags
from spriteworld import demo_ui, demo_goal_ui

FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'spriteworld.configs.protoge.goal_general',
                    'Module name of task config to use.')
flags.DEFINE_string('mode', 'train', 'Task mode, "train" or "test"]')
flags.DEFINE_boolean('task_hsv_colors', True,
                     'Whether the task config uses HSV as color factors.')
flags.DEFINE_boolean('simple_config', True,
                     'Whether to use simple config.')
flags.DEFINE_integer('render_size', 256,
                     'Height and width of the output image.')
flags.DEFINE_integer('anti_aliasing', 10, 'Renderer anti-aliasing factor.')
flags.DEFINE_integer('level', 0,
                     'Difficulty level for the config.')



from spriteworld import environment, renderers, sprite, tasks, action_spaces
import os, copy

GOAL_ENV_IMAGE_RENDERERS = {
  'observation': renderers.PILRenderer((100,100)),
  'achieved_goal': renderers.AchievedPILGoalRenderer((100,100)),
  'desired_goal': renderers.PILGoalRenderer((100, 100))
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
    'renderers': GOAL_ENV_IMAGE_RENDERERS,
    'init_sprites': init_sprites,
    'max_episode_length': 1000,
    'metadata': {
        'name': os.path.basename(__file__)
    }
}
simple_config = {
    'task': tasks.SparseGoalPlacement(),
    'action_space': action_spaces.Navigate(),
    'renderers': GOAL_ENV_IMAGE_RENDERERS,
    'init_sprites': simple_sprites,
    'max_episode_length': 1000,
    'metadata': {
        'name': os.path.basename(__file__)
    }
}


def main(_):
  config = importlib.import_module(FLAGS.config)
  config = config.get_config(mode=FLAGS.mode, level=FLAGS.level)
  demo_goal_ui.setup_run_ui(config, FLAGS.render_size, False,
                       FLAGS.anti_aliasing)

  # if FLAGS.simple_config:
  #   demo_goal_ui.setup_run_ui(simple_config, FLAGS.render_size, False,
  #                      FLAGS.anti_aliasing)
  # else:
  #   demo_goal_ui.setup_run_ui(test_config, FLAGS.render_size, False,
  #                      FLAGS.anti_aliasing)
  #demo_ui.setup_run_ui(config, FLAGS.render_size, FLAGS.task_hsv_colors,
  #                     FLAGS.anti_aliasing)


if __name__ == '__main__':
  app.run(main)
