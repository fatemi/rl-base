import logging
import numpy as np
from copy import deepcopy
import pygame

logger = logging.getLogger(__name__)

from IPython.core import debugger
debug = debugger.Pdb().set_trace


# RGB colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Catch(object):
    def __init__(self, grid_size=10, length=1, basket_offset=1, frame_skip=1, window=5, noisy=False,
                 rendering=False, rendering_scale=15):
        self.grid_size = grid_size
        self.noisy = noisy
        self.length = length
        self.basket_offset = basket_offset
        self.play = 0
        self.state = None
        self.state_shape = [grid_size, grid_size]
        self.frame_skip = frame_skip
        self.nb_actions = 3
        self.window = window
        self._rendering = rendering
        self.rendering_scale = rendering_scale
        self.done = False
        self.reset()

    @property
    def rendering(self):
        return self._rendering

    @rendering.setter
    def rendering(self, flag):
        if flag is True:
            if self._rendering is False:
                self._init_pygame()
                self._rendering = True
        else:
            self.close()
            self._rendering = False

    def reset(self):
        self.done = False
        n = int(np.random.randint(0, self.grid_size-1, size=1))
        m = int(np.random.randint(1, self.grid_size-2, size=1))
        self.state = np.asarray([0, n, m])
        self.play = 0
        if self.rendering:
            self._init_pygame()

    def _init_pygame(self):
        pygame.init()
        size = [self.rendering_scale * self.state_shape[0], self.rendering_scale * self.state_shape[1]]
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Catch Game")

    def close(self):
        if self.rendering:
            pygame.quit()

    def _update_state(self, action):
        """
        Input: action and states
        Output: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        elif action == 2:
            action = 1  # right
        else:
            raise ValueError('Unexpected action: {0}'.format(action))
        f0, f1, basket = state
        new_basket = min(max(self.basket_offset, basket + action), self.grid_size-self.basket_offset-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,) * 2
        state = self.state
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2] - self.basket_offset:state[2] + self.basket_offset + 1] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= self.basket_offset:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0] == self.grid_size - 1:
            self.play += 1
        if self.play >= self.length:
            self.done = True
            return True
        else:
            return False

    def get_frame(self):
        canvas = self._draw_state()
        canvas = np.copy(canvas)
        # canvas *= 255.
        canvas = canvas.reshape((1, -1))[0]
        return canvas

    def get_observations(self):
        state = self.get_frame()
        if self.noisy:
            state = (state + 0.5 * self.last_frame) / 2.0
        return state.reshape((self.grid_size, self.grid_size))

    def get_state(self):
        return self.get_observations()

    def step(self, action):
        if self._is_over():
            logger.warning('Calling step on a finished episode.')
            return self.get_observations(), None, True
        reward = 0
        if self.frame_skip == 1:
            self.last_frame = self.get_frame()
        for i in range(self.frame_skip):
            if not self._is_over():
                self._update_state(action)
                reward += self._get_reward()
            if i == self.frame_skip - 2 and self.noisy:
                self.last_frame = self.get_frame()

        new_state = self.get_observations()
        return new_state, reward, self._is_over(), {}

    def get_lives(self):
        return 0 if self.state[0] == self.grid_size-1 else 1

    def get_paddlexy_ballxy(self):
        fruit_row, fruit_col, basket = self.state
        fruit_row_ = deepcopy(fruit_row)
        if fruit_row == 0:
            fruit_row_ = self.grid_size - 1
        offset = self.window if self.window % 2 == 0 else self.window + 1
        return deepcopy(basket), deepcopy(self.grid_size) - int(offset), deepcopy(fruit_col), deepcopy(fruit_row_)

    def render(self):
        if not self.rendering or self.state is None:
            return
        pygame.event.pump()
        self.screen.fill(BLACK)
        state = self.state
        basket_size = [self.rendering_scale, self.rendering_scale * (2 * self.basket_offset + 1)]
        basket_pos = [self.rendering_scale * (self.state_shape[1] - 1),
                      self.rendering_scale * (state[2] - self.basket_offset)]
        agent = pygame.Rect(basket_pos[0], basket_pos[1], basket_size[0], basket_size[1])
        pygame.draw.rect(self.screen, WHITE, agent)
        fruit_size = [self.rendering_scale, self.rendering_scale]
        fruit = pygame.Rect(self.rendering_scale * state[0], self.rendering_scale * state[1], fruit_size[0], fruit_size[1])
        pygame.draw.rect(self.screen, WHITE, fruit)
        pygame.display.flip()


if __name__ == '__main__':
    grid_size = 24
    offset = 1  # hence, basket length == 3
    length = 1
    env = Catch(grid_size, length, offset, rendering=True)
    env.render()
    while not env.done:
        action = None
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                if event.key == pygame.K_DOWN:
                    action = 2
                if event.key == pygame.K_LEFT:
                    action = 1
                if event.key == pygame.K_RIGHT:
                    action = 1
        if action is None:
            continue
        print('action >> ', action)
        s, r, term, _ = env.step(action)
        print(' | r: ', r, ' | game_over: ', term)
        env.render()
