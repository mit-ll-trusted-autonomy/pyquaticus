from collections import OrderedDict
import gym
from inputs import get_gamepad
import math
import numpy as np
import threading
import time
import tkinter as tk

from gym_aquaticus.envs.config import WestPointConfig
from gym_aquaticus.envs.aquaticus_team_env import AquaticusTeamEnv
from gym_aquaticus.utils import vec_to_mag_heading, mag_heading_to_vec

class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self, env, max_speed):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        # specific to aquaticus
        self.env = env
        # remember last heading in case not pressing joystick
        self.prev_heading = 270
        self.partner_modes = ['DEFEND_E', 'ATTACK_MED']
        self.partner_mode_id = 0

        self._max_speed = max_speed

        self._paused = False
        self._exit = False


    def read(self): # return the buttons/triggers that you care about in this method
        x = self.LeftJoystickX
        y = self.LeftJoystickY
        a = self.A
        bX = self.X # b=1, x=2
        rb = self.RightBumper

        if bX:
            self.partner_mode_id = (self.partner_mode_id + 1) % len(self.partner_modes)
            self.X = 0 # reset the button
        return [x, y, a, bX, rb]


    def raw_action(self):
        x, y, _, bX, _ = self.read()
        self.process_pause()
        left_joystick_mag = math.sqrt(x**2 + y**2)
        left_joystick_angle = math.atan2(y, x)
        # align with heading
        if left_joystick_mag <= 5e-2:
            left_joystick_heading = self.prev_heading
        else:
            left_joystick_heading = math.degrees((left_joystick_angle + math.pi/2.0) % (2*math.pi))
        self.prev_heading = left_joystick_heading
        if bX:
            # also change the behavior of the partner
            return left_joystick_mag, left_joystick_heading, self.partner_modes[self.partner_mode_id]
        else:
            return left_joystick_mag, left_joystick_heading


    def action(self):
        act = self.raw_action()
        mag, heading = act[:2]
        mag = min(self._max_speed, 2*self._max_speed*mag)
        if len(act) > 2:
            return [np.array([mag, heading], dtype=np.float32), act[2]]
        else:
            return [np.array([mag, heading], dtype=np.float32)]


    def trigger_pause(self):
        self._paused = True
        self.process_pause(newgame=True)


    def process_pause(self, newgame=False):
        if not self._paused:
            return

        win = tk.Tk()
        #Set the geometry of Tkinter frame
        win.geometry("750x270")
        win.title('Menu')
        if newgame:
            tk.Label(win, text="New Game", font=('Helvetica 14 bold')).pack(pady=20)
            tk.Label(win, text="Press Start to begin. To exit press B.", font=('Helvetica 14 bold')).pack(pady=20)
        else:
            tk.Label(win, text="Paused...", font=('Helvetica 14 bold')).pack(pady=20)
            tk.Label(win, text="Press Start to resume. To quit press B.", font=('Helvetica 14 bold')).pack(pady=20)

        while self._paused and not self._exit:
            #Create an instance of Tkinter frame
            win.update()
            self.env.pause()
            time.sleep(0.5)
        win.destroy()

        if self._exit:
            raise RuntimeError('EXIT')
        else:
            self.env.unpause()

    def _monitor_controller(self):
        last_btn = []
        while True:
            events = get_gamepad()
            event_codes = [e.code for e in events]
            if len(events) == 1 and events[0].code == 'SYN_REPORT':
                # ignore synchronization report
                continue
            if event_codes == last_btn:
                # every press seems to be duplicated
                last_btn = []
                continue
            elif 'BTN' in event_codes[0]:
                last_btn = event_codes
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.X = event.state
                elif event.code == 'BTN_WEST':
                    self.Y = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                    if self._paused:
                        self._exit = True
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                    self._paused = not self._paused
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state

if __name__ == '__main__':
    config_dict = {
        'moos_config': WestPointConfig(),
        'red_team_params': [('localhost', 9011, 'red_one')],
        'blue_team_params': [('localhost', 9015, 'blue_one')],
        'sim_script': './demo_1v1.sh',
        'frame': 'world',
        'team': 'red',
        'single_agent': 'red_one',
        'shoreside_params': ('localhost', 9000, 'shoreside'),
        'return_raw_state': True
    }
    env = AquaticusTeamEnv(config_dict)

    # env = gym.make('gym_aquaticus:aquaticus-v0', sim_script='./demo_1v1.sh', verbose=0, config_version='popolopen', perpetual=True)
    env.reset()
    joy = XboxController(env, env._moos_config.speed_bounds[1])
    done = False
    total_reward = 0.0
    action_dict = OrderedDict()
    while not done:
        action = joy.action()
        # helper functions to convert between magnitude, heading and x/y vector
        # vec = mag_heading_to_vec(action[0][0], action[0][1])
        # mag, hdg = vec_to_mag_heading(vec)
        # action[0] = np.array([mag, hdg], dtype=np.float32)
        obs, reward, done, info = env.step(action[0])
        total_reward += reward
        print(f'Reward: {reward}')

    print(f'Episode ended with reward: {total_reward}')
    env.close()
