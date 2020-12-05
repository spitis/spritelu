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
"""Interactive GUI for Spriteworld.

Be aware that this UI overrides the action space and renderer for ease of
playing, so those will be different from what are specified in the task config.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as log
import sys
from absl import logging
from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import time

from spriteworld import action_spaces
from spriteworld import environment
from spriteworld import renderers


class MatplotlibUI(object):
  """Class for visualising the environment based on Matplotlib."""

  def __init__(self):
    self.rewards = 10 * [np.nan]
    self.rewards_bounds = [-1, 0]
    self.last_success = None

    plt.ion()
    self._fig = plt.figure(
        figsize=(18, 12), num='Spriteworld', facecolor='white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1,1])
    self._ax_image = plt.subplot(gs[0,0])
    self._ax_image.axis('off')

    self._ax_goal = plt.subplot(gs[0, 1])
    self._ax_goal.axis('off')

    self._ax_scalar = plt.subplot(gs[1, 0])
    self._ax_scalar.spines['right'].set_visible(False)
    self._ax_scalar.spines['top'].set_visible(False)
    self._ax_scalar.xaxis.set_ticks_position('bottom')
    self._ax_scalar.yaxis.set_ticks_position('left')
    self._setup_callbacks()

  @property
  def ax_image(self):
    return self._ax_image

  @property
  def ax_goal(self):
    return self._ax_goal

  def _setup_callbacks(self):
    """Default callbacks for the UI."""

    # Pressing escape should stop the UI
    def _onkeypress(event):
      if event.key == 'escape':
        # Stop UI
        logging.info('Pressed escape, stopping UI.')
        plt.close(self._fig)
        sys.exit()

    self._fig.canvas.mpl_connect('key_release_event', _onkeypress)

    # Disable default keyboard shortcuts
    for key in ('keymap.fullscreen', 'keymap.home', 'keymap.back',
                'keymap.forward', 'keymap.pan', 'keymap.zoom', 'keymap.save',
                'keymap.quit', 'keymap.grid', 'keymap.yscale', 'keymap.xscale',
                'keymap.all_axes'):
      plt.rcParams[key] = ''

    # Disable logging of some matplotlib events
    log.getLogger('matplotlib').setLevel('WARNING')

  def _draw_observation(self, image, action):
    """Draw the latest observation."""
    self._ax_image.clear()
    self._ax_image.imshow(255 - image, interpolation='none')
    self._ax_image.set_xticks([])
    self._ax_image.set_yticks([])
    if action is not None:
      self._ax_image.annotate(
          '',
          xycoords='axes fraction',
          xy=action[:2],  # Start of arrow
          xytext=action[2:],  # End of arrow
          arrowprops={
              'arrowstyle': '<|-',
              'color': 'red',
              'lw': 4,
          })

    # Indicate success
    linewidth = 1
    color = 'black'
    if np.isnan(self.rewards[-1]):
      linewidth = 8
      color = 'green' if self.last_success else 'red'

    for sp in self._ax_image.spines.values():
      sp.set_color(color)
      sp.set_linewidth(linewidth)

  def _draw_goal(self, image):
    """Draw the latest observation."""
    self._ax_goal.clear()
    self._ax_goal.imshow(255 - image, interpolation='none')
    self._ax_goal.set_xticks([])
    self._ax_goal.set_yticks([])

    # Indicate success
    linewidth = 1
    color = 'black'
    for sp in self._ax_goal.spines.values():
      sp.set_color(color)
      sp.set_linewidth(linewidth)

  def _draw_rewards(self):
    """Draw the past rewards plot."""
    self._ax_scalar.clear()
    self._ax_scalar.set_ylabel('Rewards')
    self._ax_scalar.set_xlabel('Timestep')
    xs = np.arange(-len(self.rewards), 0)
    self._ax_scalar.set_xticks(xs)
    self._ax_scalar.axhline(y=0.0, color='lightgrey', linestyle='--')
    self._ax_scalar.stem(xs, self.rewards, basefmt=' ')

    self._ax_scalar.set_xlim((xs[0] - 1.0, xs[-1] + 1.0))
    self._ax_scalar.set_ylim(
        (self.rewards_bounds[0] - 1.0, self.rewards_bounds[1] + 1.0))

  def register_callback(self, event_name, callback):
    """Register a callback for the given event."""
    self._fig.canvas.mpl_connect(event_name, callback)

  def update(self, timestep, action):
    """Update the visualisation with the latest timestep and action."""
    reward = timestep.reward
    if reward is None:
      reward = np.nan
    self.rewards = self.rewards[1:] + [reward]
    self.rewards_bounds[0] = np.nanmin(
        [np.nanmin(self.rewards), self.rewards_bounds[0]])
    self.rewards_bounds[1] = np.nanmax(
        [np.nanmax(self.rewards), self.rewards_bounds[1]])

    
    self._draw_observation(timestep.observation['observation'], action)
    self._draw_goal(timestep.observation['desired_goal'])
    self._draw_rewards()
    plt.show(block=False)

    self.last_success = timestep.observation['success']


class HumanNavigateAction(object):
  """Demo agent for mouse-clicking interface with Navigate action space."""

  def __init__(self, action_space, timeout=600):
    self._action_space = action_space
    self._click = None
    self._timeout = timeout

  def help(self):
    logging.info('Click to start move, then click again to select direction+magnitude.')

  def register_callbacks(self, ui):
    """Register the matplotlib callbacks required by the agent."""

    def _onclick(event):
      if event.inaxes and event.inaxes == ui.ax_image:
        # Map the click into axis-fraction positions (origin at bottom-left).
        self._click = event.inaxes.transAxes.inverted().transform((event.x, event.y))
      else:
        self._click = None
      return

    def _onrelease(event):
      self._click = None
      return

    def _onmove(event):
      if self._click is not None:
        if event.inaxes and event.inaxes == ui.ax_image:
          # Map the click into axis-fraction positions (origin at bottom-left).
          self._click = event.inaxes.transAxes.inverted().transform((event.x, event.y))
        else:
          self._click = None
      return

    ui.register_callback('button_release_event', _onrelease)
    ui.register_callback('button_press_event', _onclick)
    ui.register_callback('motion_notify_event', _onmove)

  def begin_episode(self):
    logging.info('Starting episode')

  def step(self, timestep):
    """Take a step."""
    del timestep  # Unused

    def _get_click():
      """Get mouse click."""
      click = self._click
      if click is not None:
        plt.waitforbuttonpress(timeout=0.01)
      click = self._click
      while click is None:
        x = plt.waitforbuttonpress(timeout=self._timeout)
        if x is None:
          logging.info('Timed out. You took longer than %d seconds to click.', self._timeout)
        elif x:
          logging.info('You pressed a key, but were supposed to click with the mouse.')
          self.help()
        else:
          click = self._click
      return click

    def _get_action():
      """Get action from user."""
      click_to = _get_click()
      try:
        action = np.array(click_to).astype(np.float32)

        if any(np.isnan(action)):
          raise ValueError
        self._action_space.action_spec().validate(action)
        return action
      except (ValueError, TypeError):
        logging.info('Select a valid action')
        return _get_action()

    action = _get_action()
    return action

def setup_run_ui(env_config, render_size, task_hsv_colors, anti_aliasing):
  """Start a Demo UI given an env_config."""
  assert isinstance(env_config['action_space'], action_spaces.Navigate)
  agent = HumanNavigateAction(env_config['action_space'])

  env_config['renderers'] = {
      'observation':
          renderers.PILRenderer(
              image_size=(render_size, render_size),
              anti_aliasing=anti_aliasing),
      'desired_goal':
          renderers.PILGoalRenderer(
              image_size=(render_size, render_size),
              anti_aliasing=anti_aliasing),
      'success':
          renderers.Success()
  }
  env = environment.Environment(**env_config)
  ui = MatplotlibUI()
  agent.register_callbacks(ui)

  # Start RL loop
  timestep = env.reset()
  ui.update(timestep, action=None)

  while True:
    target = agent.step(timestep)
    action = np.clip((target - env._sprites[-1].position)*4., -1., 1.)
    logging.info('Taking action: {}'.format(action))
    timestep = env.step(action)
    ui.update(timestep, np.concatenate((env._sprites[-1].position, target)))