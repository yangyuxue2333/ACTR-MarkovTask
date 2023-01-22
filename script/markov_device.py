## ================================================================ ##
## MARKOV_DEVICE.PY                                                        ##
## ================================================================ ##
## A simple ACT-R device for the MARKOV task                        ##
## -----------------------------------------                        ##
## This is a device that showcases the unique capacities of the new ##
## JSON-RPC-based ACT-R interface. The device is written in Python, ##
## and interacts with ACT-R entirely through Python code.           ##
## The Simon task is modeled after Andrea Stocco's (2016)          ##
## paper on the Simon task.                           ##
## ================================================================ ##



import os
from typing import Dict

import actr
import random
import numpy as np
import pandas as pd
from pathlib import Path
import pprint as p
import json
import time
from datetime import datetime
from functools import reduce
import scipy.optimize as opt
import itertools
from pandas import CategoricalDtype

STATE = (0, 1, 2)
TEXT = ("", "A1", "A2", "B1", "B2", "C1", "C2")
COLOR = ("GREEN", "RED", "BLUE")
LOCATION = ("LEFT", "RIGHT")
RESPONSE_MAP = [{'f':'A1', 'k':'A2'},
                {'f':'B1', 'k':'B2'},
                {'f':'C1', 'k':'C2'}]

# ACTR PARAMETERS
ACTR_PARAMETER_NAMES = ['v', 'seed', 'ans', 'lf', 'bll',  'mas', 'egs', 'alpha', 'imaginal-activation']

# TASK PARAMETERS
M = 1
REWARD: Dict[str, float] = {'B1': 2, 'B2': 2, 'C1': 2, 'C2': 2}
PROBABILITY = {'MARKOV_PROBABILITY':.7, 'REWARD_PROBABILITY': {'B1': 0.26, 'B2': 0.57, 'C1': 0.41, 'C2': 0.28}}
RANDOM_WALK = True  #Enable random walk for reward probability
RANDOM_TABLE, RANDOM_NOISE_TABLE = None, None

random.seed(0)


class MarkovStimulus:
    """An abstract Markov task stimulus"""

    def __init__(self, text="A1"):
        """
        <[MARKOV_STIMULUS]'A1' state: 1; color: GREEN at: [LEFT>]
        """
        assert (text in TEXT)
        self.kind = "MARKOV_STIMULUS"
        self.state = ""
        self.text = text
        self.color = ""
        self.location = ""
        self.init_stimulus()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, val):
        self._text = val

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, val):
        self._color = val

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, val):
        self._location = val

    def init_stimulus(self):
        """
        given text stimulus, automatically generate state, color, and location
        """
        if self.text == "":
            self.state = 0
        elif self.text == "A1":
            self.state = 1
            self.color = 'GREEN'
            self.location = 'LEFT'
        elif self.text == "A2":
            self.state = 1
            self.color = 'GREEN'
            self.location = 'RIGHT'
        elif self.text == "B1":
            self.state = 2
            self.color = 'RED'
            self.location = 'LEFT'
        elif self.text == "B2":
            self.state = 2
            self.color = 'RED'
            self.location = 'RIGHT'
        elif self.text == "C1":
            self.state = 2
            self.color = 'BLUE'
            self.location = 'LEFT'
        elif self.text == "C2":
            self.state = 2
            self.color = 'BLUE'
            self.location = 'RIGHT'
        else:
            print('error')

    def __str__(self):
        return "<[%s]'%s' state: %s; color: %s at: [%s>]" % (
        self.kind, self.text, self.state, self.color, self.location)

    def __repr__(self):
        return self.__str__()


class MarkovState():
    """A class for a Markov State"""

    def __init__(self):
        """Inits a markov state
        <[MARKOV_STATE] state:'0'; '(A1, A2)'; response:[None][None]; response_time:0.0>
        """

        self.kind = "MARKOV_STATE"
        self.state = 0
        self.markov_probability = PROBABILITY['MARKOV_PROBABILITY']
        self.reward_probability_fixed = PROBABILITY['REWARD_PROBABILITY'] # fixed based on init reward probability
        self.reward_probability_random_walk = {'B1': 0, 'B2': 0, 'C1': 0, 'C2': 0}  # slowly update reward probability

        self.reward_dict = REWARD
        self.state1_stimuli = None
        self.state2_stimuli = None

        self.state1_response = None
        self.state2_response = None
        self.state1_response_time = 0.0
        self.state2_response_time = 0.0
        self.state1_selected_stimulus = None
        self.state2_selected_stimulus = None
        self.received_reward = 0

        self.state_frequency = None
        self.reward_frequency = None


        ### ACTR TRACE
        self.actr_chunk_trace = None
        self.actr_production_trace = None
        self.state0()

    @property
    def state1_response_time(self):
        return self._state1_response_time

    @property
    def state2_response_time(self):
        return self._state2_response_time

    @property
    def reward_probability_random_walk(self):
        return self._reward_probability_random_walk

    @state1_response_time.setter
    def state1_response_time(self, val):
        self._state1_response_time = val

    @state2_response_time.setter
    def state2_response_time(self, val):
        self._state2_response_time = val

    @reward_probability_random_walk.setter
    def reward_probability_random_walk(self, val):
        self._reward_probability_random_walk = val

    def get_letter_frequency(self, probability):
        if probability > .5:
            return 'common'
        else:
            return 'rare'

    def state0(self):
        self.state1_stimuli = MarkovStimulus('A1'), MarkovStimulus('A2')
        self.state = 0

    def state1(self, response):

        self.state1_response = response
        self.state1_selected_stimulus = RESPONSE_MAP[0][response]

        if RESPONSE_MAP[0][response] == 'A1':
            # use pseudo_random_numbers() rather than random.random() to fix the randomness
            if MarkovACTR.pseudo_random() < self.markov_probability:
                self.state2_stimuli = MarkovStimulus('B1'), MarkovStimulus('B2')
                # log reward frequency
                #self.state_frequency = 'common'
                self.state_frequency = self.get_letter_frequency(self.markov_probability)
            else:
                self.state2_stimuli = MarkovStimulus('C1'), MarkovStimulus('C2')
                #self.state_frequency = 'rare'
                self.state_frequency = self.get_letter_frequency(1-self.markov_probability)
            self.state2_selected_stimulus = RESPONSE_MAP[1][response]

        if RESPONSE_MAP[0][response] == 'A2':
            if MarkovACTR.pseudo_random() < self.markov_probability:
                self.state2_stimuli = MarkovStimulus('C1'), MarkovStimulus('C2')
                #self.state_frequency = 'common'
                self.state_frequency = self.get_letter_frequency(self.markov_probability)
            else:
                self.state2_stimuli = MarkovStimulus('B1'), MarkovStimulus('B2')
                #self.state_frequency = 'rare'
                self.state_frequency = self.get_letter_frequency(1-self.markov_probability)
            self.state2_selected_stimulus = RESPONSE_MAP[2][response]

        self.state = 1

    def state2(self, response):
        self.state2_response = response
        left, right = self.state2_stimuli
        if left.text == 'B1' and right.text == 'B2':
            self.state2_selected_stimulus = RESPONSE_MAP[1][response]

        if left.text == 'C1' and right.text == 'C2':
            self.state2_selected_stimulus = RESPONSE_MAP[2][response]
        self.state = 2

    def reward(self):
        """
        self.reward_probability = 'REWARD_PROBABILITY': {'B1': 0.26, 'B2': 0.57, 'C1': 0.41, 'C2': 0.28}
        self.reward_probability_random_walk = {'B1':0, 'B2':0, 'C1':0, 'C2':0} (init values)
            random walk values should be updated after each trial
        self.reward_dict = {'B1':2, 'B2':2, 'C1':2,'C2':2}

        self.curr_reward_probability <- updated reward_pobability value
        self.curr_reward_probability_dict <- updated reward_probability_dict

        """
        # enable random walk
        if (RANDOM_WALK is True):
            self.curr_reward_probability = self.reward_probability_random_walk[self.state2_selected_stimulus] # num value
            self.curr_reward_probability_dict = self.reward_probability_random_walk # dict
        else:
            # log reward probability
            self.curr_reward_probability = self.reward_probability_fixed[self.state2_selected_stimulus]
            self.curr_reward_probability_dict = self.reward_probability_fixed

        # decide received reward
        if MarkovACTR.pseudo_random() < self.curr_reward_probability: #self.curr_reward_probability[self.state2_selected_stimulus]:
            self.received_reward = self.reward_dict[self.state2_selected_stimulus]
            # log reward frequency
            self.reward_frequency = self.get_letter_frequency(self.curr_reward_probability)
        else:
            self.received_reward = 0  # default reward value
            # log reward frequency
            self.reward_frequency = self.get_letter_frequency(1-self.curr_reward_probability)

        # print('TEST L281: self.curr_reward_probability; self.reward_frequency', self.curr_reward_probability, self.reward_frequency, )

        # update state
        self.state = 3

    # =================================================== #
    # ACTR CHUNK TRACE INFO
    # =================================================== #
    @property
    def actr_chunk_names(self):
        """
        return a list of DM chunk names
        ['M1-1', 'M1-2'...]
        """

        res = []
        for i in ['A', 'B', 'C']:
            for j in range(1, 5):
                res.append('M' + '-' + i + str(j))
        self._actr_chunk_names = res
        return self._actr_chunk_names

    @property
    def actr_production_names(self):
        """
        return a list of production names
        """
        self._actr_production_names = ['CHOOSE-STATE1-LEFT', 'CHOOSE-STATE1-RIGHT', 'CHOOSE-STATE2-LEFT', 'CHOOSE-STATE2-RIGHT']
        return self._actr_production_names

    @staticmethod
    def actr_production_parameter(production_name, parameter_name):
        """
        Extract the parameter value of a production during model running
            NOTE: must be called during actr running
        """
        assert (production_name in actr.all_productions() and
                parameter_name in [':u', ':utility', ':at', ':reward', ':fixed-utility'])
        actr.hide_output()
        value = actr.spp(production_name, parameter_name)[0][0]
        actr.unhide_output()
        return (production_name, parameter_name, value)

    @staticmethod
    def actr_chunk_parameter(chunk_name, parameter_name):
        """
        Extract the parameter value of a chunk during model running
            NOTE: must be called during actr running
        """
        actr.hide_output()
        try:
            value = actr.sdp(chunk_name, parameter_name)[0][0]
            actr.unhide_output()
        except:
            print('ERROR: WRONG', chunk_name, parameter_name)
            value = None
        finally:
            actr.unhide_output()
            return (chunk_name, parameter_name, value)

    def get_actr_chunk_trace(self, parameter_name=':Last-Retrieval-Activation'):
        """
        Return a list chunk's info (e.g. ":Last-Retrieval-Activation", ":Activation"
            NOTE: must be called during actr running
        """
        return [self.actr_chunk_parameter(c, parameter_name)[-1] for c in self.actr_chunk_names]

    def get_actr_production_trace(self, parameter_name=':utility'):
        """ Return a list chunk's info (e.g. ":Last-Retrieval-Activation", ":Activation"
            NOTE: must be called during actr running
        """
        return [self.actr_production_parameter(p, parameter_name)[-1] for p in self.actr_production_names]

    def __str__(self):
        return "<[%s] \t[%s, %.2f]'%s' \t[%s, %.2f]'%s' \tR:[%s] \t[%s][%s]" % (
            self.kind,
            self.state1_response,
            self.state1_response_time,
            self.state1_selected_stimulus,
            self.state2_response,
            self.state2_response_time,
            self.state2_selected_stimulus,
            self.received_reward,
            self.state_frequency.upper()[0],
            self.reward_frequency.upper()[0])

    def __repr__(self):
        return self.__str__()


class MarkovACTR(MarkovState):
    def __init__(self, setup=False):
        self.index = 0
        self.log = []
        self.onset = 0.0
        self.offset = 0.0

        if setup:
            self.setup()

        # log ACTR trace
        # if log_full_trace is enabled, every trial's actr trace will be stored
        # if disabled, only last trial's actr trace will be stored
        self.log_full_trace = False

        # init markov state
        self.markov_state = None

        # init parameters
        self.actr_parameters = {}
        self.task_parameters = {}

        # init pseudo_random_table
        self.init_pseudo_random_tables()


    def setup(self,
              model="markov-model1",
              actr_params=None,
              task_params=None,
              reload=True,
              verbose=False):
        # TODO: add function to modify parameter set
        self.model = model
        self.verbose = verbose


        # init working dir
        script_dir = os.path.join(os.path.dirname(os.path.realpath('../__file__')), 'script')
        # print('test curr_dir', script_dir)


        self.add_actr_commands()
        if reload:
            # schedule event of detect production/reward before loading model
            # note: if not load, no need to schedule it again
            # actr.schedule_event_now("detect-produc            # actr.schedule_event_now("detect-production-hook")tion-hook")
            # actr.schedule_event_now("detect-reward-hook")

            # load model
            actr.load_act_r_model(os.path.join(script_dir, "markov-core.lisp"))
            actr.load_act_r_model(os.path.join(script_dir, model + ".lisp"))
            actr.add_dm(*self.actr_dm())
            actr.load_act_r_model(os.path.join(script_dir, "markov-memory.lisp"))

        # init parameter sets
        self.actr_parameters = self.get_default_actr_parameters()
        self.task_parameters = self.get_default_task_parameters()

        # update parameter sets
        self.set_actr_parameters(actr_params)
        self.set_task_parameters(task_params)

        # goal focus
        mot = str(self.task_parameters['M'] * np.max(list(self.task_parameters['REWARD'].values())))
        actr.define_chunks(['start-trial', 'isa', 'phase', 'step', 'attend-stimulus', 'motivation', mot, 'time-onset', '0.0', 'previous-reward', '0.0', 'current-reward', '0.0'])
        actr.goal_focus('start-trial')

        window = actr.open_exp_window("MARKOV TASK", width=500, height=250, visible=False)
        self.window = window

        actr.install_device(window)

        if verbose: print(self.__str__())

    @property
    def response_time(self):
        return self.offset - self.onset

    def add_actr_commands(self):
        actr.add_command("markov-update-state0", self.update_state0, "Update window: fixation")
        actr.add_command("markov-update-state1", self.update_state1, "Update window: state1")
        actr.add_command("markov-update-state2", self.update_state2, "Update window: state2")
        actr.add_command("markov-update-state3", self.update_state3, "Update window: state3")
        actr.add_command("markov-key-press", self.respond_to_key_press, "markov task output-key monitor")
        actr.monitor_command("output-key", "markov-key-press")
        actr.add_command("markov-update-end", self.update_end, "Update window: done")

        actr.add_command("markov-update-reward-probability", self.update_random_walk_reward_probabilities, "Update reward probability: before state0")

    def remove_actr_commands(self):
        actr.remove_command("markov-update-state0")
        actr.remove_command("markov-update-state1")
        actr.remove_command("markov-update-state2")
        actr.remove_command("markov-update-state3")
        actr.remove_command("markov-key-press")
        actr.remove_command("markov-update-end")

        actr.remove_command("markov-update-reward-probability")

    def respond_to_key_press(self, model, key):
        self.response = key
        actr.clear_exp_window()
        self.offset = actr.mp_time()

        # print('previous state', self.markov_state.state,
        #      'response', key,
        #      'rt: ', self.response_time)

        # imortant step to proceed task
        # by calling next_step()
        self.next_state()

    def next_state(self):
        '''decide next state based on response
           self.markov_state will be updated based on response

           e.g. self.markov_state.state == 0, then call self.markov_state.state1(key)
           e.g. self.markov_state.state == 2, then call self.markov_state.state2(key)
                                              then call self.markov_state.reward()
           note: the amount of reward is specified by REWARD dict()

           this function will be called in respond_to_key_press() in state 1, 2
           and in update_state3() in state 3
        '''

        if self.markov_state.state == 0:
            self.markov_state.state1(self.response)
            self.markov_state.state1_response_time = self.response_time
            actr.schedule_event_relative(.01, "markov-update-state2")
        else:
            self.markov_state.state2(self.response)
            self.markov_state.state2_response_time = self.response_time

            # continue deliver rewards, no need to wait for response
            self.markov_state.reward()
            actr.schedule_event_relative(.01, "markov-update-state3")

        # log
        if self.markov_state.state == 3:

            # log actr trace
            # if self.log_full_trace is enabled, every trials' trace will be stored
            # if not, only last trial or all trials
            if self.log_full_trace:
                self.markov_state.actr_chunk_trace = {':Activation':self.markov_state.get_actr_chunk_trace(parameter_name=':Activation'),
                                                      ':Last-Retrieval-Activation':self.markov_state.get_actr_chunk_trace(parameter_name=':Last-Retrieval-Activation'),
                                                      ':Reference-Count':self.markov_state.get_actr_chunk_trace(parameter_name=':Reference-Count')}
                self.markov_state.actr_production_trace = self.markov_state.get_actr_production_trace(parameter_name=':utility')

            self.log.append(self.markov_state)

            if self.verbose:
                print(self.markov_state)

    def update_state0(self):
        # self.markov_state = MarkovState()
        actr.clear_exp_window()
        actr.add_text_to_exp_window(self.window, '+', x=200, y=100, color='black', font_size=50)

    def update_state1(self):
        actr.clear_exp_window()
        state1 = actr.add_visicon_features(
            ['isa', ['markov-stimulus-location', 'markov-stimulus'],
             'kind', 'markov-stimulus',
             'stage', 1,
             'state', 'A',
             'color', 'green',
             'left-stimulus', 'A1',
             'right-stimulus', 'A2',
             'screen-x', 200, 'screen-y', 100])
        self.onset = actr.mp_time()

    def update_state2(self):
        # print('test: update_state2()',
        #      self.markov_state.state2_stimuli[0].text,
        #      self.markov_state.state2_stimuli[1].text)
        actr.clear_exp_window()
        state2 = actr.add_visicon_features(
            ['isa', ['markov-stimulus-location', 'markov-stimulus'],
             'kind', 'markov-stimulus',
             'stage', 2,
             'state', self.markov_state.state2_stimuli[0].text[0],
             'color', 'orange',
             'left-stimulus', self.markov_state.state2_stimuli[0].text,
             'right-stimulus', self.markov_state.state2_stimuli[1].text,
             'screen-x', 200, 'screen-y', 100])
        self.onset = actr.mp_time()

    def update_state3(self):
        # print('test: update_state3()',
        #       self.markov_state.received_reward)
        actr.clear_exp_window()
        R = actr.add_visicon_features(
            ['isa', ['markov-stimulus-location', 'markov-reward'],
             'kind', 'markov-reward',
             'stage', 3,
             'reward', self.markov_state.received_reward,
             'screen-x', 200, 'screen-y', 100])

    def update_end(self):
        actr.clear_exp_window()
        actr.add_text_to_exp_window(self.window, 'done', x=200, y=100, color='black', font_size=30)

    def update_random_walk_reward_probabilities(self):
        """
        This function enables random walk reward probabilities at the start of each trial

        access previous state from log, and current state
        calculate random walk probability for current state
        update random walk probability for
        """
        # init markov state
        self.markov_state = MarkovState()

        # calculate random walk noise
        if len(self.log) > 0:
            pre_state = self.log[-1]
            curr_state = self.markov_state
            pre_state_p_dict = pre_state.reward_probability_random_walk
            curr_state_p_dict = curr_state.reward_probability_random_walk

            for key, prob in pre_state_p_dict.items():
                # use pseudo_random_noise() rather than random.normal
                #prob_rw = prob + np.random.normal(loc=0, scale=0.025)
                prob_rw = prob + MarkovACTR.pseudo_random_noise()
                while (prob_rw < 0.25) | (prob_rw > 0.75):
                    #prob_rw = prob + np.random.normal(loc=0, scale=0.025)
                    prob_rw = prob + MarkovACTR.pseudo_random_noise()
                # update current state's reward probability based on random walk algorithm
                curr_state_p_dict[key] = prob_rw
            # update to state properties
            self.markov_state.reward_probability_random_walk = curr_state_p_dict
        else:
            self.markov_state.reward_probability_random_walk = self.markov_state.reward_probability_fixed.copy()
        # print('test after reward_probability_random_walk', self.markov_state.reward_probability_random_walk)


    def run_experiment(self, n=2):
        """
        run ACT-R model
            m = MarkovACTR(setup=False)
            m.setup(reload=True)
            m.run_experiment(2)
        """

        for i in range(n):
            actr.schedule_event_relative(.01 + 100 * i, "markov-update-reward-probability")  # random walk
            actr.schedule_event_relative(1 + 100 * i, "markov-update-state0")
            actr.schedule_event_relative(2 + 100 * i, "markov-update-state1")

            self.index += 1

        # "done"
        actr.schedule_event_relative(80 + 100 * i, "markov-update-end")
        actr.run(81 + 100 * i)

        # clear commands
        self.remove_actr_commands()

    # =================================================== #
    # PARAMETER SETUP
    # =================================================== #

    def get_actr_parameter(self, param_name):
        """
        get parameter from current model
        :param keys: string, the parameter name (e.g. ans, bll, r1, r2)
        :return:
        """
        assert param_name in ACTR_PARAMETER_NAMES
        return actr.get_parameter_value(":" + param_name)

    def get_actr_parameters(self, *kwargs):
        param_set = {}
        for param_name in kwargs:
            param_set[param_name] = self.get_actr_parameter(param_name)
        return param_set

    def set_actr_parameters(self, kwargs):
        """
        set parameter to current model
        :param kwargs: dict pair, indicating the parameter name and value (e.g. ans=0.1, r1=1, r2=-1)
        :return:
        """
        # print("start assign set_parameters", kwargs)
        # print('before', self.actr_parameters)
        actr.hide_output()
        update_parameters = self.actr_parameters.copy()
        # if new para given
        if kwargs:
            update_parameters.update(kwargs)
            for key, value in kwargs.items():
                actr.set_parameter_value(':' + key, value)
            self.actr_parameters = update_parameters

        # if no new param given
        else:
            self.actr_parameters = self.get_default_actr_parameters()
        # self.actr_parameters["seed"] = str(self.actr_parameters["seed"])
        actr.unhide_output()
        # print('after', self.actr_parameters)

    def set_task_parameters(self, kwargs):
        """
        Set the task parameter
             self.task_parameter = {'MARKOV_PROBABILITY': 0.7,
                                    'REWARD_PROBABILITY': 0.7,
                                    'REWARD': {'B1': 2, 'B2': 0, 'C1': 0, 'C2': 0
                                    'RANDOM_WALK': True}
        If kwargs == None: return default parameters
        """
        # print('before', self.task_parameters)
        # new = self.task_parameters.copy()
        # if new para given

        global M
        global REWARD
        global PROBABILITY
        global RANDOM_WALK

        # init default parameters
        self.task_parameters = self.get_default_task_parameters()

        # pass default params to global variables
        M = self.task_parameters['M']
        REWARD = self.task_parameters['REWARD']
        PROBABILITY = {'MARKOV_PROBABILITY':self.task_parameters['MARKOV_PROBABILITY'],
                       'REWARD_PROBABILITY':self.task_parameters['REWARD_PROBABILITY']}
        RANDOM_WALK = self.task_parameters['RANDOM_WALK']

        # update new params to global variables
        if kwargs:
            # print('before update', REWARD, PROBABILITY)
            for key, value in kwargs.items():
                # print(key, value)
                if key == 'M':
                    M = value
                if key == 'REWARD':
                    REWARD = value
                if key == 'MARKOV_PROBABILITY':
                    PROBABILITY['MARKOV_PROBABILITY'] = value
                if key == 'REWARD_PROBABILITY':
                    PROBABILITY['REWARD_PROBABILITY'] = value
                if key == 'RANDOM_WALK':
                    RANDOM_WALK = value
            # print('after update', REWARD, PROBABILITY)
            self.task_parameters = {**PROBABILITY, 'REWARD': REWARD, 'RANDOM_WALK':RANDOM_WALK, 'M':M}
        # print('after', self.task_parameters)

    # =================================================== #
    # PSEUDO RANDOMNESS
    # =================================================== #
    def init_pseudo_random_tables(self):
        """
        Create a random numer table
        """
        np.random.seed(0)
        n = 1000
        # use infinite iterator
        # when it reaches to end, will return from back
        global RANDOM_TABLE
        global RANDOM_NOISE_TABLE
        RANDOM_TABLE = itertools.cycle(np.random.random_sample(n).tolist())
        RANDOM_NOISE_TABLE = itertools.cycle(np.random.normal(loc=0, scale=0.025, size=n).tolist())

    @staticmethod
    def pseudo_random():
        """
        To alleviate the impact of randomness, we use pseudo random generator
        Each random number generated during the model simulation is accessed
        from a pre-generated random number table.
        """
        # print('access random number')
        # return random.random()
        return next(RANDOM_TABLE)

    @staticmethod
    def pseudo_random_noise():
        """
        To alleviate the impact of randomness, we use pseudo random generator
        Each random number generated during the model simulation is accessed
        from a pre-generated random number table.
        """
        # print('access random noise')
        # return np.random.normal(loc=0, scale=0.025)
        return next(RANDOM_NOISE_TABLE)

    # =================================================== #
    # DM SETUP
    # =================================================== #
    def actr_dm(self):
        """
        Add declarative memory
        CHUNK0-0
           STATUS  PROCESS
           LEFT-STIMULUS  A1
           RIGHT-STIMULUS  A2
           CURR-STATE  A
           NEXT-STATE  B
           RESPONSE  RIGHT
        """

        global REWARD
        reward = [max(REWARD.values()), 0]
        comb1 = [i for i in itertools.product(['A'], ['LEFT', 'RIGHT'], ['B', 'C'], ['none'])]
        comb2 = [i for i in itertools.product(['B', 'C'], ['LEFT', 'RIGHT'], ['none'], reward)]
        comb = comb1 + comb2
        dm = []
        i = 0
        for c in comb:
            s, a, s_, r = c[0], c[1], c[2], c[3]
            i += 1
            if i > 4: i = 1  # 4 * 3 memory
            #print(s, a, s_, r)
            name = 'M' + '-' + s + str(i)
            dm.append([name, 'isa', 'wm',
                       'status', 'process',
                       'left-stimulus', s+'1',
                       'right-stimulus', s+'2',
                       'curr-state', s,
                       'next-state', s_,
                       'response', a,
                       'reward', r])
        return dm

    def get_default_actr_parameters(self):
        """
        default act-r parameter sets

        """
        return self.get_actr_parameters(*ACTR_PARAMETER_NAMES)

    def get_default_task_parameters(self):
        """
        default parameter sets
        """
        return {'M': 1,
                'MARKOV_PROBABILITY': 0.7,
                'REWARD_PROBABILITY': {'B1': 0.26, 'B2': 0.57, 'C1': 0.41, 'C2': 0.28},
                'REWARD': {'B1': 2, 'B2': 2, 'C1': 2, 'C2': 2},
                'RANDOM_WALK': True}


    # =================================================== #
    # STATS
    # =================================================== #

    def df_behaviors(self):
        """
        Return model generated beh data
        """
        rows = [[
            s.state1_response,
            s.state1_response_time,
            s.state1_selected_stimulus,
            s.state2_response,
            s.state2_response_time,
            s.state2_selected_stimulus,
            s.received_reward,
            s.state_frequency,
            s.reward_frequency,
            self.task_parameters['M']
        ] for s in self.log]

        df = pd.DataFrame(rows, columns= ['state1_response',
                                            'state1_response_time',
                                            'state1_selected_stimulus',
                                            'state2_response',
                                            'state2_response_time',
                                            'state2_selected_stimulus',
                                            'received_reward',
                                            'state_frequency',
                                            'reward_frequency',
                                            'm_parameter']).reset_index()
        df['pre_received_reward'] = df['received_reward'].shift()
        df['pre_received_reward'] = df.apply(lambda x: x['pre_received_reward'] if  pd.isnull(x['pre_received_reward'])\
            else ('non-reward' if x['pre_received_reward']==0 else 'reward'), axis=1)
        return df

    def calculate_stay_probability(self):
        """
        Calculate the probability of stay:
            A trial is marked as "STAY" if the agent selects the same action in current trial (e.g. LEFT)
            as the previous trial

        NOTE: will be -1 trials because exclude NA rows
        """

        df = self.df_behaviors()
        df['state1_stay'] = df['state1_response'].shift() # first row is NA (look at previsou trial)
        df['state1_stay'] = df.apply(
            lambda x: 1 if x['state1_stay'] == x['state1_response'] else (np.nan if pd.isnull(x['state1_stay']) else 0), axis=1)
        # df['pre_received_reward'] = df['received_reward'].shift()
        # df['pre_received_reward'] = df.apply(lambda x: 'non-reward' if x['pre_received_reward'] == 0 else 'reward', axis=1)
        df = df.dropna(subset=['state1_stay', 'pre_received_reward'])
        df = df.astype({'state_frequency': CategoricalDtype(categories=['common', 'rare'], ordered=True),
                        'pre_received_reward': CategoricalDtype(categories=['reward', 'non-reward'], ordered=True)})
        df = df[['index', 'state_frequency', 'received_reward', 'pre_received_reward', 'state1_stay', 'state1_response_time', 'state2_response_time']]
        return df

    def calculate_real_frequency(self, merge=False):
        """
        Calculate the real markov transition frequency
            if merge = True, return full DF with real transition frequency
            if merge = False, return only real transition frequency table

            state1_frequency = the number of selecting A1/total num of trials
            state2_frequency1 = proportion of A1 -> B; A1 -> C
            state2_frequency2 = proportion of A2 -> C; A2 -> B
        """
        df = self.df_behaviors()
        df['state1_selected_stimulus_type'] = df.apply(lambda x: x['state1_selected_stimulus'][0], axis=1)
        df['state2_selected_stimulus_type'] = df.apply(lambda x: x['state2_selected_stimulus'][0], axis=1)

        df1 = pd.DataFrame(df.value_counts(['state1_selected_stimulus'], normalize=True),
                           columns=['state1_frequency']).reset_index()
        df2 = pd.DataFrame(df[df['state1_selected_stimulus'] == 'A1'].value_counts(
            ['state1_selected_stimulus', 'state2_selected_stimulus_type'], normalize=True),
            columns=['state2_frequency1']).reset_index()
        df3 = pd.DataFrame(df[df['state1_selected_stimulus'] == 'A2'].value_counts(
            ['state1_selected_stimulus', 'state2_selected_stimulus_type'], normalize=True),
            columns=['state2_frequency2']).reset_index()
        res = df1.merge(df2, how='outer').merge(df3, how='outer')
        if merge:
            res = df.merge(df1, how='left').merge(df2, how='left').merge(df3, how='left').round(2)
        return res

    def df_reward_probabilities(self, format='long'):
        """
        Return reward probabilities if random walk is enabled
        """
        assert format in ('wide', 'long')
        # log curr reward probability
        # if random walk is enabled, curr_reward_probability_dict is reward_probability_random_walk
        # otherwise, curr_reward_probability_dict is reward_probability_fixed
        df_wide = pd.DataFrame([{'state2_selected_stimulus': s.state2_selected_stimulus,
                                 'received_reward': s.received_reward,
                                 **s.curr_reward_probability_dict} for s in self.log]).reset_index()

        df_long = df_wide.melt(id_vars=['index', 'state2_selected_stimulus', 'received_reward'],
                           var_name='state2_stimulus', value_name='reward_probability').sort_values(by='index')
        if format == 'long':
            res = df_long
        else:
            res = df_wide

        return res

    def df_optimal_behaviors(self, num_bin=4):
        """
        Estimate the optimal response
        First, look at selected responses from state2, and curr_reward_probability_dict.
        If random walk is enabled, curr_reward_probability_dict = reward_probability_random_walk
        Otherwise, curr_reward_probability_dict = reward_probability_fixed
        Second, bin trials into 4 blocks (default), and calculate the mean reward probability in each block
        Third, find out the state2 response with hightest reward probability, this is the optimal response
        Lastly, join the optimal response back to wide df
        """
        df_wide = pd.DataFrame([{'state1_selected_stimulus': s.state1_selected_stimulus,
                                 'state2_selected_stimulus': s.state2_selected_stimulus,
                                 'received_reward': s.received_reward,
                                 **s.curr_reward_probability_dict} for s in self.log]).reset_index()
        df_wide['index_bin'] = pd.cut(df_wide['index'], num_bin, labels=False, ordered=False, right=False)
        df_max = df_wide.groupby(['index_bin'])[['B1', 'B2', 'C1', 'C2']].mean().reset_index()

        # estimate the optimal state2 response by highest reward probabilities
        df_max['state2_optimal'] = df_max[['B1', 'B2', 'C1', 'C2']].idxmax(axis=1)

        # estimate the optimal state1 response by common path from state1 to state2
        df_max['state1_optimal'] = df_max['state2_optimal'].replace({'B1': 'A1', 'B2': 'A1', 'C1': 'A2', 'C2': 'A2'})

        res = pd.merge(df_wide[['index', 'index_bin']], df_max[['index_bin', 'state1_optimal', 'state2_optimal']],
                       how='left')
        return res

    def df_postprocess_behaviors(self):
        """
        Return post processed beh data: each row is one trial

        1. reward probabilities for four state2 stimulus
        2. model simulated beh data
        3. estimated optimal response. This is done by binned trials.
            For example, 100 trials are binned to 4 blocks. In each block, the optimal state2 response
            is determined by the averaged highest reward probability. Correspondinglt, the optimal state1
            response is estimated by common path from state1 to state2. If optimal state is B2, then optimal
            state1 should be A1. If optimal state is C2, then optimal tate1 should be A2.
        """

        # merge reward probabilities
        # wide format of reward probabilities
        dfr = self.df_reward_probabilities(format='wide').rename({'B1':'reward_probability_B1',
                                                               'B2':'reward_probability_B2',
                                                               'C1':'reward_probability_C1',
                                                               'C2':'reward_probability_C2'}, axis=1)
        # dfr_max = dfr.loc[dfr.groupby(['index'])['reward_probability'].idxmax(axis=0)]
        # merge beh data, optimal beh estimation and reward probabilities
        df = pd.merge(pd.merge(self.df_behaviors(), self.df_optimal_behaviors()), dfr, how='left')

        df['received_reward_norm'] = df['received_reward'] / df['received_reward'].max()
        df['received_reward_sum'] = df['received_reward_norm'].cumsum()
        df['is_optimal'] = df.apply(lambda x: 1 if (x['state1_selected_stimulus'] == x['state1_optimal'] and x['state2_selected_stimulus'] == x['state2_optimal']) else 0, axis=1)

        # instead of using fixed optimal response(fixed reward), we estimate optimal by binned trials into 4 blocks
        # response2 associated with the highest reward probabilities (binned) are optimal response
        # response1 is estimated from common frequency (common response 1 - response 2 is optimal)
        #df['is_optimal'] = df.apply(lambda x: 1 if (x['state1_response'] == state1_response and x['state2_response'] == state2_response) else 0, axis=1)
        df['received_reward_sum_prop'] = df.apply(lambda x: x['received_reward_sum'] / ((x['index'] + 1)), axis=1)

        # calculate the cumulative optimal to estimaye learning outcome
        df = pd.merge(df, df.groupby(['index_bin'])['is_optimal'].mean().reset_index(), how='left', on='index_bin', suffixes=('', '_mean'))
        df['optimal_response_sum'] = df['is_optimal'].cumsum()
        df['optimal_response_sum_prop'] = df.apply(lambda x: x['optimal_response_sum'] / ((x['index'] + 1)), axis=1)
        return df


    def df_actr_chunk_traces(self, parameter_name=':Last-Retrieval-Activation'):
        """
        Return a chunk activation trace
        """
        df = pd.DataFrame([s.actr_chunk_trace[parameter_name] for s in self.log], columns=self.actr_chunk_names).reset_index()
        return pd.melt(df, id_vars='index', var_name='memory', value_name=parameter_name)

    def df_actr_production_traces(self, parameter_name=':utility', norm=True):
        """
        Return a production utility trace
        """
        df = pd.DataFrame([s.actr_production_trace for s in self.log], columns=self.actr_production_names).reset_index()
        df = pd.melt(df, id_vars='index', var_name='action', value_name=parameter_name)
        if norm:
            df[':utility'] = (df[':utility'] - df[':utility'].min()) / (df[':utility'].max() - df[':utility'].min())
        return df


    def df_postprocess_actr_traces(self):
        """
        Return the actr traces
            if log_full_trace == True, return full trace df
            else: return last trial's trace
        """

        if self.log_full_trace:
            # production trace
            df1 = self.df_actr_production_traces(parameter_name=':utility')
            df1['index_bin'] = pd.cut(df1['index'], 10, labels=False, ordered=False, right=False)
            df1['state'] = df1.apply(lambda x: 'STATE1' if ('CHOOSE-STATE1' in x['action']) else 'STATE2', axis=1)
            df1['response'] = df1.apply(lambda x: 'LEFT' if ('LEFT' in x['action']) else 'RIGHT', axis=1)
            df1 = df1.fillna(0.0)

            # memory trace
            try:
                df21 = self.df_actr_chunk_traces(parameter_name=':Reference-Count')
                df22 = self.df_actr_chunk_traces(parameter_name=':Last-Retrieval-Activation').fillna(0.0)
                df2 = pd.merge(df21, df22)
                df2['index_bin'] = pd.cut(df2['index'], 10, labels=False)
                df2.replace(to_replace=[None], value=np.nan, inplace=True)
                # df2['memory_type'] = df2.apply(lambda x: 'non-reward' if 'M0' in x['memory'] else 'reward', axis=1)
                df2['memory_type'] = 'non-reward'
                df2.loc[df2['memory'].str.endswith(('B1', 'B3', 'C1', 'C3')), 'memory_type'] = 'reward'
            except:
                print('no memory trace')
                df2 = None
            finally:
                df1 = df1.astype({'index_bin': float, ':utility': float})
                df2 = df2.astype({'index_bin': float, ':Reference-Count': float, ':Last-Retrieval-Activation': float})
                return df1, df2

        else:
            actr.hide_output()
            # production trace
            df1 = pd.DataFrame(np.array([self.actr_production_names,
                                         [actr.spp(p, ':utility')[0][0] for p in self.actr_production_names]]).T,
                               columns=['action', ':utility']).astype({':utility': float}).reset_index()
            df1['state'] = df1.apply(lambda x: 'STATE1' if ('CHOOSE-STATE1' in x['action']) else 'STATE2', axis=1)
            df1['response'] = df1.apply(lambda x: 'LEFT' if ('LEFT' in x['action']) else 'RIGHT', axis=1)

            # memory trace
            #chunk_names = ['M-A1', 'M-A2', 'M-A3', 'M-A4', 'M-B1', 'M-B2', 'M-B3', 'M-B4', 'M-C1', 'M-C2', 'M-C3', 'M-C4']
            df2 = pd.DataFrame(np.array([self.actr_chunk_names,
             [actr.sdp(t, ':Reference-Count')[0][0] for t in self.actr_chunk_names],
             [actr.sdp(t, ':Activation')[0][0] for t in self.actr_chunk_names],
             [actr.sdp(t, ':Last-Retrieval-Activation')[0][0] for t in self.actr_chunk_names]]).T,
             columns=['memory', ':Reference-Count', ':Activation', ':Last-Retrieval-Activation']).astype({':Reference-Count': float,
                     ':Activation': float, ':Last-Retrieval-Activation': float}).reset_index()

            df2.replace(to_replace=[None], value=np.nan, inplace=True)
            df2['memory_type'] = 'non-reward'
            df2.loc[df2['memory'].str.endswith(('B1', 'B3', 'C1', 'C3')), 'memory_type'] = 'reward'
            return df1, df2

    def __str__(self):
        header = "######### SETUP MODEL " + self.model + " #########"
        task_parameter_info = ">> TASK PARAMETERS: " + str(self.task_parameters) + " <<"
        actr_parameter_info = ">> ACT-R PARAMETERS: " + str(self.actr_parameters) + " <<"
        return "%s\n \t%s\n \t%s\n" % (header, task_parameter_info, actr_parameter_info)

    def __repr__(self):
        return self.__str__()


class MarkovHuman(MarkovState):
    """A class for recording a Markov trial"""

    def __init__(self, model='markov-monkey', verbose = True):
        """Inits a markov trial
        stimuli contain a list of markov states, e.g. [("A1", "A2"), ("B1", "B2")]
        """
        self.kind = model
        self.index = 0

        self.log = []
        self.verbose = verbose

        # init markov state
        self.markov_state = None
        self.action_space = ['f', 'k']
        self.response = None

        # init parameters
        self.rl_parameters = {}
        self.task_parameters = {'MARKOV_PROBABILITY': 0.7,
                                'REWARD_PROBABILITY': {'B1': 0.26, 'B2': 0.57, 'C1': 0.41, 'C2': 0.28},
                                'REWARD': {'B1': 2, 'B2': 2, 'C1': 2, 'C2': 2}}

    def respond_to_key_press(self):
        key = None
        if self.kind == 'markov-monkey':
            key = self.response_monkey()

        elif self.kind == 'markov-left':
            key = self.response_left()

        else:
            key = self.action_space[1]

        self.response = key
        self.next_state(key)

    def response_monkey(self):
        return random.choice(self.action_space)

    def response_left(self):
        return self.action_space[0]

    def next_state(self, response):
        '''decide next state based on response
           self.markov_state will be updated based on response

           e.g. self.markov_state.state == 0, then call self.markov_state.state1(key)
           e.g. self.markov_state.state == 2, then call self.markov_state.state2(key)
                                              then call self.markov_state.reward()
           note: the amount of reward is specified by REWARD dict()

           this function will be called in respond_to_key_press() in state 1, 2
           and in update_state3() in state 3
        '''
        # print('test', self.markov_state.state1_stimuli, self.markov_state.state2_stimuli)
        if self.markov_state.state == 0:
            self.markov_state.state1(response)
            self.markov_state.state1_response_time = 0.0


        else:
            self.markov_state.state2(response)
            self.markov_state.state2_response_time = 0.0

            # continue deliver rewards, no need to wait for response
            self.markov_state.reward()

        # log
        if self.markov_state.state == 3:

            # log actr trace
            self.log.append(self.markov_state)
            if self.verbose:
                print(self.markov_state)

    def run_experiment(self, n=1):
        """
        """
        for i in range(n):
            self.markov_state = MarkovState()
            self.markov_state.state0()  # init
            self.respond_to_key_press()
            self.respond_to_key_press()
            self.index += 1

    def df_behaviors(self):
        """
        Return model generated beh data
        """
        rows = [[
            s.state1_response,
            s.state1_response_time,
            s.state1_selected_stimulus,
            s.state2_response,
            s.state2_response_time,
            s.state2_selected_stimulus,
            s.received_reward,
            s.state_frequency,
            s.reward_frequency
        ] for s in self.log]
        return pd.DataFrame(rows, columns= ['state1_response',
                                            'state1_response_time',
                                            'state1_selected_stimulus',
                                            'state2_response',
                                            'state2_response_time',
                                            'state2_selected_stimulus',
                                            'received_reward',
                                            'state_frequency',
                                            'reward_frequency'])

    def df_reward_probabilities(self):
        """
        Return reward probabilities if random walk is enabled
        """
        df_wide = pd.DataFrame([{'state2_selected_stimulus': s.state2_selected_stimulus,
                                 'received_reward': s.received_reward,
                                 **s.reward_probability_random_walk} for s in self.log])
        df_wide = df_wide.reset_index()
        res = df_wide.melt(id_vars=['index', 'state2_selected_stimulus', 'received_reward'],
                           var_name='state2_stimulus', value_name='reward_probability').sort_values(by='index')
        return res

    def df_optimal_behaviors(self, num_bin=4):
        """
        Estimate the optimal response
        First, look at selected responses from state2, and curr_reward_probability_dict.
        If random walk is enabled, curr_reward_probability_dict = reward_probability_random_walk
        Otherwise, curr_reward_probability_dict = reward_probability_fixed
        Second, bin trials into 4 blocks (default), and calculate the mean reward probability in each block
        Third, find out the state2 response with hightest reward probability, this is the optimal response
        Lastly, join the optimal response back to wide df
        """
        df_wide = pd.DataFrame([{'state1_selected_stimulus': s.state1_selected_stimulus,
                                 'state2_selected_stimulus': s.state2_selected_stimulus,
                                 'received_reward': s.received_reward,
                                 **s.curr_reward_probability_dict} for s in self.log]).reset_index()
        df_wide['index_bin'] = pd.cut(df_wide['index'], num_bin, labels=False, ordered=False, right=False)
        df_max = df_wide.groupby(['index_bin'])[['B1', 'B2', 'C1', 'C2']].mean().reset_index()

        # estimate the optimal state2 response by highest reward probabilities
        df_max['state2_optimal'] = df_max[['B1', 'B2', 'C1', 'C2']].idxmax(axis=1)

        # estimate the optimal state1 response by common path from state1 to state2
        df_max['state1_optimal'] = df_max['state2_optimal'].replace({'B1': 'A1', 'B2': 'A1', 'C1': 'A2', 'C2': 'A2'})

        res = pd.merge(df_wide[['index', 'index_bin']], df_max[['index_bin', 'state1_optimal', 'state2_optimal']], how='left')
        return res

    def calculate_stay_probability(self):
        """
        Calculate the probability of stay:
            A trial is marked as "STAY" if the agent selects the same action in current trial (e.g. LEFT)
            as the previous trial
        """
        df = self.df_behaviors()
        df['state1_stay'] = df['state1_response'].shift(-1)
        df['state1_stay'] = df.apply(
            lambda x: 1 if x['state1_stay'] == x['state1_response'] else (np.nan if pd.isnull(x['state1_stay']) else 0), axis=1)
        return df

    def df_postprocess_behaviors(self, state1_response='f', state2_response='k'):
        df = pd.merge(self.df_behaviors(), self.df_optimal_behaviors())
        # df['index_bin'] = pd.cut(df['index'], 10, labels=False, ordered=False, right=False)
        df['received_reward_norm'] = df['received_reward'] / df['received_reward'].max()
        df['received_reward_sum'] = df['received_reward_norm'].cumsum()
        df['is_optimal'] = df.apply(lambda x: 1 if (x['state1_response'] == x['state1_optimal'] and x['state2_response'] == x['state2_optimal']) else 0, axis=1)
        #df['is_optimal'] = df.apply(lambda x: 1 if (x['state1_response'] == state1_response and x['state2_response'] == state2_response) else 0, axis=1)
        df['received_reward_sum_prop'] = df.apply(lambda x: x['received_reward_sum'] / ((x['index'] + 1)), axis=1)
        df = pd.merge(df, df.groupby(['index_bin'])['is_optimal'].mean().reset_index(), how='left', on='index_bin', suffixes=('', '_mean'))
        df['optimal_response_sum'] = df['is_optimal'].cumsum()
        df['optimal_response_sum_prop'] = df.apply(lambda x: x['optimal_response_sum'] / ((x['index'] + 1)), axis=1)
        return df

    def __str__(self):
        header = "######### SETUP MODEL " + self.kind + " #########\n" + str(self.task_parameters)
        return header

    def __repr__(self):
        return self.__str__()


