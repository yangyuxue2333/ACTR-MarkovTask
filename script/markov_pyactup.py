## ================================================================ ##
## MARKOV_IBL.PY                                                 ##
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
import pyactup as pau
import random
from copy import copy
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import pandas as pd
import sys
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import logit, glm
import statsmodels.api as sm
from scipy import stats
from pandas import CategoricalDtype
import pprint
from scipy.special import expit
import scipy.optimize as opt
from tqdm.auto import tqdm
import glob
from scipy.stats import norm

SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname('../__file__')), 'script')
sys.path.insert(0, SCRIPT_PATH)

from markov_simulate_test import Plot, MaxLogLikelihood


TEXT = ("", "A1", "A2", "B1", "B2", "C1", "C2")
RESPONSE_MAP = [{'f': 'A1', 'k': 'A2'},
                {'f': 'B1', 'k': 'B2'},
                {'f': 'C1', 'k': 'C2'}]
RESPONSE_CODE = {'f': 'L', 'k': 'R'}
RANDOM_TABLE = None
RANDOM_NOISE_TABLE = None
REWARD_DICT = {'B1': (1,-1), 'B2': (1,-1), 'C1': (1,-1), 'C2': (1,-1)}
# Common transitions
COMMON_TRANS = {
    'f': 'B', # f to B state is common
    'k': 'C', # k to C state is common
}

# parameter names
RL_PARAMETER_NAMES = ['alpha', 'beta', 'beta_mf', 'beta_mb', 'lambda_parameter', 'p_parameter']
IBL_PARAMETER_NAMES = ['temperature', 'decay', 'beta_mf', 'beta_mb', 'lambda_parameter', 'p_parameter']
RT_PARAMETER_NAMES = ['lf', 'fixed_cost']

PARAMETER_NAMES = ['alpha', 'beta', 'beta_mf', 'beta_mb', 'lf', 'fixed_cost', 'lambda_parameter', 'p_parameter', 'temperature', 'decay', 'lf']
TASK_PARAMETER_NAMES = ['MARKOV_PROBABILITY','REWARD_PROBABILITY', 'REWARD']

# all model names
MODEL_NAMES = ['markov-rl-mf', 'markov-rl-mb', 'markov-rl-hybrid', 'markov-ibl-mb', 'markov-ibl-hybrid']


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
        self.markov_probability = 0.7
        self.curr_reward_probability = 0.0
        self.curr_reward_probability_dict = REWARD_DICT

        self.reward_dict = REWARD_DICT
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
    def state(self):
        return self._state

    @state.setter
    def state(self, val):
        self._state = val

    @property
    def state1_response_time(self):
        return self._state1_response_time

    @state1_response_time.setter
    def state1_response_time(self, val):
        self._state1_response_time = val

    @property
    def state2_response_time(self):
        return self._state2_response_time

    @state2_response_time.setter
    def state2_response_time(self, val):
        self._state2_response_time = val

    @property
    def state1_q(self):
        return self._state1_q

    @state1_q.setter
    def state1_q(self, val):
        self._state1_q = val

    @property
    def state2_q(self):
        return self._state2_q

    @state2_q.setter
    def state2_q(self, val):
        self._state2_q = val

    def get_letter_frequency(self, probability):
        if probability > .5:
            return 'common'
        else:
            return 'rare'

    def state0(self):
        self.state1_stimuli = MarkovStimulus('A1'), MarkovStimulus('A2')
        self.state = 0
        self._curr_state = 'A'
        self._curr_stage = '1'

    def state1(self, response):

        self.state1_response = response
        self.state1_selected_stimulus = RESPONSE_MAP[0][response]

        if RESPONSE_MAP[0][response] == 'A1':
            # use pseudo_random_numbers() rather than random.random() to fix the randomness
            if random.random() < self.markov_probability:
                self.state2_stimuli = MarkovStimulus('B1'), MarkovStimulus('B2')
                # log reward frequency
                # self.state_frequency = 'common'
                self.state_frequency = self.get_letter_frequency(self.markov_probability)
                self._curr_state = 'B'
                self._curr_stage = '2'
            else:
                self.state2_stimuli = MarkovStimulus('C1'), MarkovStimulus('C2')
                # self.state_frequency = 'rare'
                self.state_frequency = self.get_letter_frequency(1 - self.markov_probability)
                self._curr_state = 'C'
                self._curr_stage = '2'
            self.state2_selected_stimulus = RESPONSE_MAP[1][response]

        if RESPONSE_MAP[0][response] == 'A2':
            if random.random() < self.markov_probability:
                self.state2_stimuli = MarkovStimulus('C1'), MarkovStimulus('C2')
                # self.state_frequency = 'common'
                self.state_frequency = self.get_letter_frequency(self.markov_probability)
                self._curr_state = 'C'
                self._curr_stage = '2'
            else:
                self.state2_stimuli = MarkovStimulus('B1'), MarkovStimulus('B2')
                # self.state_frequency = 'rare'
                self.state_frequency = self.get_letter_frequency(1 - self.markov_probability)
                self._curr_state = 'B'
                self._curr_stage = '2'
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
        self.curr_reward_probability = self.curr_reward_probability_dict[self.state2_selected_stimulus]

        # decide received reward
        if random.random() < self.curr_reward_probability:  # self.curr_reward_probability[self.state2_selected_stimulus]:
            self.received_reward = self.reward_dict[self.state2_selected_stimulus][0]
            # log reward frequency
            self.reward_frequency = self.get_letter_frequency(self.curr_reward_probability)
        else:
            self.received_reward = self.reward_dict[self.state2_selected_stimulus][1]  # default reward
            # log reward frequency
            self.reward_frequency = self.get_letter_frequency(1 - self.curr_reward_probability)

        # update state
        self.state = 3


    def state1_predetermined(self, response, next_state, state_frequency):
        """
        Pre determined next state
        :param response:
        :param next_state:
        :return:
        """

        self.state1_response = response
        self.state1_selected_stimulus = RESPONSE_MAP[0][response]
        if next_state == 'B':
            self.state2_stimuli = MarkovStimulus('B1'), MarkovStimulus('B2')
            # log reward frequency
            # self.state_frequency = 'common'
            self.state_frequency = state_frequency
            self._curr_state = 'B'
            self._curr_stage = '2'
        elif next_state == 'C':
            self.state2_stimuli = MarkovStimulus('C1'), MarkovStimulus('C2')
            # self.state_frequency = 'rare'
            self.state_frequency = state_frequency
            self._curr_state = 'C'
            self._curr_stage = '2'
        else:
            print('Error', next_state)
            pass
        self.state2_selected_stimulus = RESPONSE_MAP[1][response]
        self.state = 1

    def state2_predetermined(self, response):
        self.state2_response = response
        left, right = self.state2_stimuli
        if left.text == 'B1' and right.text == 'B2':
            self.state2_selected_stimulus = RESPONSE_MAP[1][response]

        if left.text == 'C1' and right.text == 'C2':
            self.state2_selected_stimulus = RESPONSE_MAP[2][response]
        self.state = 2

    def reward_predtermined(self, received_reward, reward_frequency):
        """
        """
        self.received_reward = received_reward
        self.reward_frequency = reward_frequency
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
        assert (len(self._actr_chunk_names) > 0)
        return self._actr_chunk_names

    @property
    def actr_production_names(self):
        """
        return a list of production names
        """
        self._actr_production_names = ['CHOOSE-STATE1-LEFT',
                                       'CHOOSE-STATE1-RIGHT',
                                       'CHOOSE-STATE2-LEFT',
                                       'CHOOSE-STATE2-RIGHT']
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
            RESPONSE_CODE[self.state1_response],
            self.state1_response_time,
            self.state1_selected_stimulus,
            RESPONSE_CODE[self.state2_response],
            self.state2_response_time,
            self.state2_selected_stimulus,
            self.received_reward,
            self.state_frequency.upper()[0],
            self.reward_frequency.upper()[0])

    def __repr__(self):
        return self.__str__()


class MarkovIBL(MarkovState):
    """A class for recording a Markov trial"""

    def __init__(self, model='markov-rl-mf', verbose=True, **params):
        """Inits a markov trial
        stimuli contain a list of markov states, e.g. [("A1", "A2"), ("B1", "B2")]
        """
        assert (model in ('markov-ibl-mb', 'markov-ibl-hybrid', 'markov-rl-mf', 'markov-rl-mb', 'markov-rl-hybrid'))
        self.kind = model
        self.index = 0

        self.log = []
        self.verbose = verbose

        # init markov state
        self.markov_state = None
        self.action_space = ['f', 'k']
        self.state_space = ['A', 'B', 'C']
        self.stage_space = ['1', '2']
        self.response = None

        # init pseudo_random_table
        self.init_pseudo_random_tables()

        # init state
        self.markov_state = MarkovState()
        # init IBL memory
        self.memory = pau.Memory(**params)
        self.init_memory()
        self.init_parameters()

        self.q = {(s, a): 0 for s in self.state_space for a in self.action_space}
        self.p = {(s, a): 0 for s in self.state_space for a in self.action_space}
        if self.kind == 'markov-rl-hybrid':
            self.q_mf = {(s, a): 0 for s in self.state_space for a in self.action_space}
            self.q_mb = {(s, a): 0 for s in self.state_space for a in self.action_space}

        self.LL = 0.0

        # trial duration
        self.advance_time = 50

    # =================================================== #
    # SETUP
    # =================================================== #
    def init_memory(self):
        """Initializes the agent with some preliminary SDUs (to make the first choice possible)"""
        self.memory.reset()
        for s_ in ['B', 'C']:
            for a in self.action_space:
                for a_ in self.action_space:
                    for r1, r2 in list(self.markov_state.reward_dict.values()):
                        self.memory.learn(state='<S%d>' % (1), curr_state='A', next_state=s_, response=a, reward=r1)
                        self.memory.learn(state='<S%d>' % (1), curr_state='A', next_state=s_, response=a, reward=r2)
                        self.memory.learn(state='<S%d>' % (2), curr_state=s_, next_state=None, response=a_, reward=r1)
                        self.memory.learn(state='<S%d>' % (2), curr_state=s_, next_state=None, response=a_, reward=r2)
                        self._r1 = r1
                        self._r2 = r2
        self.memory.activation_history = []

    def init_parameters(self):

        # RL parameters ['beta', 'beta1', 'beta2', 'alpha', 'alpha1', 'alpha2', 'lambda', 'p', 'w', 'temperature', 'decay', 'lf']
        # RL MF/MB parameters
        self.alpha = .2  # learning rate
        # self.alpha1 = .2  # learning rate
        # self.alpha2 = .2  # learning rate
        self.beta = 5  # stage2 exploration rate
        self.beta_mf = 5  # stage1 mf exploration rate
        self.beta_mb = 5  # stage1 mb exploration rate

        self.temperature = .2   # temperature parameter
        self.decay = .2         # decay parameter
        self.lf = .63           # latency factor (F)
        self.fixed_cost = .585           # latency factor (F)
        self.lambda_parameter = .2  # decay rate
        self.p_parameter = 0    # perseveration rate
        # self.w_parameter = 0  # w = 0: pure MF

        # IBL parameters
        self.memory.noise = self.temperature
        self.memory.decay = self.decay


        # init task parameters
        reward_dict = REWARD_DICT
        self.task_parameters = {'MARKOV_PROBABILITY': 0.7,
                                'REWARD_PROBABILITY': 'LOAD',
                                'REWARD': reward_dict,
                                # RL Parameter
                                'alpha': self.alpha,  # learning rate
                                # 'alpha1' : self.alpha1,  # learning rate
                                # 'alpha2' : self.alpha2,  # learning rate
                                'beta': self.beta,  # exploration rate at second state
                                'beta_mf' : self.beta_mf,  # exploration rate
                                'beta_mb' : self.beta_mb,  # exploration rate
                                'lambda_parameter' : self.lambda_parameter,  # decay rate
                                'p_parameter' : self.p_parameter,  # decay rate
                                # 'w_parameter' : self.w_parameter,  # w = 0: pure MF
                                # IBL Parameter
                                'temperature' : self.temperature,  # temperature rate
                                'decay' : self.decay,           # decay rate
                                # retrieval latency parameters
                                'lf': self.lf,                  # latency factor (F)
                                'fixed_cost': self.fixed_cost   # fixed cost for time retrieval
        }

    def update_parameters(self, **kwargs):
        """
        Update parameter
        :param kwargs:
        :return:
        """
        assert (set(kwargs.keys()).issubset(set(self.task_parameters.keys())))
        self.task_parameters.update(**kwargs)

        # update RL parameters
        self.alpha = self.task_parameters['alpha']
        # self.alpha1 = self.task_parameters['alpha1']
        # self.alpha2 = self.task_parameters['alpha2']
        self.beta = self.task_parameters['beta']
        self.beta_mf = self.task_parameters['beta_mf']
        self.beta_mb = self.task_parameters['beta_mb']
        self.temperature = self.task_parameters['temperature']
        self.decay = self.task_parameters['decay']
        self.lambda_parameter = self.task_parameters['lambda_parameter']
        self.p_parameter = self.task_parameters['p_parameter']
        # self.w_parameter = self.task_parameters['w_parameter']

        # update IBL parameters
        self.memory.noise = self.temperature
        self.memory.decay = self.decay

        # retrieval latency parameters
        self.lf = self.task_parameters['lf']
        self.fixed_cost = self.task_parameters['fixed_cost']

        # update reward
        global REWARD_DICT
        REWARD_DICT = self.task_parameters['REWARD']

        if self.verbose:
            print(self.__str__())

    def init_pseudo_random_tables(self):
        """
        Create a random number table
        """
        np.random.seed(0)
        n = 1000
        # use infinite iterator
        # when it reaches to end, will return from back
        global RANDOM_TABLE
        global RANDOM_NOISE_TABLE
        RANDOM_TABLE = itertools.cycle(np.random.random_sample(n).tolist())
        RANDOM_NOISE_TABLE = itertools.cycle(np.random.normal(loc=0, scale=0.025, size=n).tolist())

        # generate random walk probability
        self._random_table = RANDOM_TABLE
        self._random_noise_table = RANDOM_NOISE_TABLE

        # load random walk probability
        self._random_walk_table = self.load_random_walk_reward_probabilities()
        self._random_walk_table_iter = itertools.cycle(self._random_walk_table)

    # =================================================== #
    # PROCEED EXPERIMENT
    # =================================================== #

    def respond_to_key_press(self):
        """
        Proceed Experiment and record response
        :return:
        """
        key = None
        if self.kind == 'markov-ibl-mb':
            key = self.choose_ibl_mb()
        elif self.kind == 'markov-ibl-hybrid':
            key = self.choose_ibl_hybrid()
        elif self.kind =='markov-rl-mf':
            key = self.choose_rl_mf()
        elif self.kind =='markov-rl-mb':
            key = self.choose_rl_mb()
        # simple mf/mb hybrid:
        elif self.kind =='markov-rl-hybrid':
            key = self.choose_rl_hybrid()
        else:
            print('error model', self.kind)

        self.response = key
        self.next_state(key)

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
            # self.markov_state.state1_response_time = 0.0


        else:
            self.markov_state.state2(response)
            # self.markov_state.state2_response_time = 0.0

            # continue deliver rewards, no need to wait for response
            self.update_random_walk_reward_probabilities()
            self.markov_state.reward()

        # log
        if self.markov_state.state == 3:
            # update q and encode memory based on algorithm
            self.encode_memory()
            self.update_q()

            # log actr trace
            self.log.append(self.markov_state)

            if self.verbose:
                print(self.markov_state)


    def update_random_walk_reward_probabilities(self):
        """
        This function enables random walk reward probabilities at the start of each trial

        access previous state from log, and current state
        calculate random walk probability for current state
        update random walk probability for
        """
        curr_reward_probability_dict = next(self._random_walk_table_iter)

        # update to state properties
        self.markov_state.curr_reward_probability_dict = curr_reward_probability_dict

    def load_random_walk_reward_probabilities(self):
        """
        Load pre-generated reward probabilities (Nussenbaum, K. et al., 2020)
        """
        data_dir = os.path.join(os.path.dirname(os.getcwd()), 'data/fixed')
        dfr = pd.read_csv(os.path.join(data_dir, 'masterprob4.csv'))
        dfr.columns = ['B1', 'B2', 'C1', 'C2']
        dict_list = dfr.to_dict('records')
        return dict_list

    # =================================================== #
    # PROCEED FROM DATA
    # =================================================== #

    def respond_from_data(self, response1, response2, received_reward, state2, state_frequency, reward_frequency):
        """
        Proceed Experiment by passing in key press
        :return:
        """
        if self.markov_state._curr_stage == '1':
            LL = self.estimate_state1_response_LL(key= response1)
            self.markov_state._LL = LL
            self.LL += LL
            self.response = response1
        # do not estimate the LL for response2
        else:
            # retrieve event and advance event
            # does not impact RL LL estimation
            self.response = response2
            self.rl_state2_choice()
            self.ibl_state2_choice()
        # proceed to next trial
        self.next_state_from_data(response1, response2, received_reward, state2, state_frequency, reward_frequency)


    def next_state_from_data(self, response1, response2, received_reward, state2, state_frequency, reward_frequency):

        # print('test', self.markov_state.state1_stimuli, self.markov_state.state2_stimuli)
        if self.markov_state.state == 0:
            self.markov_state.state1_predetermined(response=response1, next_state=state2, state_frequency=state_frequency)
            self.markov_state.state1_response_time = 0.0

        else:
            self.markov_state.state2_predetermined(response=response2)
            self.markov_state.state2_response_time = 0.0

            self.markov_state.reward_predtermined(received_reward=received_reward, reward_frequency=reward_frequency)

        if self.markov_state.state == 3:
            # update q and encode memory based on algorithm
            self.encode_memory()
            self.update_q()

            # log actr trace
            self.log.append(self.markov_state)

            # display trial
            if self.verbose:
                print(self.markov_state)

    # =================================================== #
    # ESTIMATE PARAMETER FROM DATA
    # =================================================== #

    def estimate_state1_response_LL(self, key):
        """
        Estimate the Log-Likelihood of state1 response
        :param key:
        :param prev_choice:
        :return: LL

        Required parameter:
            beta
            p_parameter
        """
        q = self.q.copy()

        # control stickness
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == key else -1

        # Estimate q based on different algorithms
        # RL
        if self.kind.startswith('markov-rl-'):
            # evaluate both
            # self.evaluate_rl_mf(response=key)
            # self.evaluate_rl_mb(response=key)
            self.evaluate_rl_hybrid(response=key)
            mf_value = self.markov_state._mf_value  # self.beta_mf * q_mf
            mb_value = self.markov_state._mb_value  # self.beta_mb * q_mb
            hybrid_value = self.markov_state._mf_value + self.markov_state._mb_value

            # RL-MF
            if self.kind == 'markov-rl-mf':
                estimated_q = mf_value

            # RL-MB
            elif self.kind == 'markov-rl-mb':
                estimated_q =  mb_value

            # RL-Hybrid
            elif self.kind == 'markov-rl-hybrid':
                estimated_q = hybrid_value
            else:
                pass

        # IBL
        elif self.kind.startswith('markov-ibl-'):
            self.evaluate_ibl_hybrid(response=key)
            mb_value = self.markov_state._mb_value
            hybrid_value = self.markov_state._hybrid_value

            # IBL-MB
            if self.kind == 'markov-ibl-mb':
                estimated_q = mb_value

            # IBL-Hybrid
            elif self.kind == 'markov-ibl-hybrid':
                estimated_q = hybrid_value
        else:
            pass

        # estimate log-likelihood
        p = expit(estimated_q)
        return np.log(max(p, 10e-10))

    # =================================================== #
    # EVALUATE STATE2
    # =================================================== #

    def evaluate_rl_mf(self, response=None):
        """
        Evaluate Q table using RL-MF. If no response is provided, always evaluate LEFT
        :return:
        """
        assert self.markov_state._curr_stage == '1'
        q = self.q.copy()

        # Q(MF)
        if not response:
            response = self.action_space[0] # always eval left
        q_mf = q[('A', response)] - q[('A', MarkovIBL.return_alternative_item(self.action_space, response))]

        # scale by free parameter beta_mf
        mf_value = self.beta_mf * q_mf

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == response else -1

        # softmax choice rule
        # p = expit(self.beta * (q[('A', self.action_space[0])] - q[('A', self.action_space[1])] + rep * self.p_parameter))
        p = expit(mf_value + rep * self.p_parameter)

        # save expecetd value of mf = beta_mb * q_value
        self.markov_state._mf_value = mf_value
        self.markov_state._state1_p = p
        return p

    def evaluate_rl_mb(self, response=None):
        """
        Evaluate Q value using RL-MB
         >> best parameter combination
            alpha=.5,
            beta=5,
            p_parameter=0,
            lambda_parameter=.6
        :return:
        """
        assert self.markov_state._curr_stage == '1'
        # Q(MF)
        if not response:
            response = self.action_space[0]  # always eval left

        q = self.q.copy()
        b_value = max([q[('B', a)] for a in self.action_space])
        c_value = max([q[('C', a)] for a in self.action_space])

        # Determine the choice
        if COMMON_TRANS[response] == 'B':
            q_mb = (2 * .7 - 1) * (b_value - c_value)
        else:
            q_mb = (2 * .7 - 1) * (c_value - b_value)

        # scale by beta_mb
        mb_value = self.beta_mb * q_mb

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == response else -1

        # softmax choice rule
        p = expit(mb_value + rep * self.p_parameter)

        # save expected q value of mb = beta_mb * q_value
        self.markov_state._mb_value = mb_value
        self.markov_state._state1_p = p
        return p

    def evaluate_rl_hybrid(self, response=None):
        """
        According to Daw 2011
        Evaluate Q table using RL-Hybrid
        >> best parameter combination
            alpha=.5,
            beta=5,
            p_parameter=0,
            lambda_parameter=.6 
            beta_mf
            beta_mb
        """
        if not response:
            response = self.action_space[0]  # always eval left

        self.evaluate_rl_mf()
        self.evaluate_rl_mb()
        # q_hybrid = self.w_parameter * self.markov_state._q_mb + (1 - self.w_parameter) * self.markov_state._q_mf

        # note: no need to x beta_mf since mf_value is already multiplied by beta_mf
        hybrid_value = self.markov_state._mf_value + self.markov_state._mb_value


        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == response else -1

        # softmax choice rule
        p = expit(hybrid_value + rep * self.p_parameter)  # p_left

        # Q(Hybrid)
        self.markov_state._state1_p = p
        self.markov_state._hybrid_value = hybrid_value
        return p

    def evaluate_ibl_mb(self, response=None):
        return self.evaluate_ibl_mb1(response=response)

    def evaluate_ibl_mb1(self, response=None):
        """
        Implemented based on to Andrea's idea: use IBL to retrieve frequency, use RL-Q learning to evaluate Q
            q_mb = diff Q value of two s_
            mb_value is scaled by beta_mb
        :param response:
        :return:
        """
        q = self.q.copy()

        # simulate response
        if not response:

            # IBL Learning: retrieve the most likely (frequent) next state given a1
            next_states = [self.start_retrieving(rehearse=False, curr_state='A', response=a1)['next_state'] for a1 in self.action_space]
            # evaluate max Q of s_
            next_state_max_q = [np.max([q[(s_, a2)] for a2 in self.action_space]) for s_ in next_states]

            # decide action based on highest Q
            max_id = np.argmax(next_state_max_q)
            a = self.action_space[max_id]
            q_mb = next_state_max_q[0] - next_state_max_q[1]  # left - right

        # estimate from data (response is known)
        else:
            a = self.action_space[0]  # always evaluate left
            s_ = self.start_retrieving(rehearse=False, curr_state='A', response=a)['next_state']
            alt_s_ = MarkovIBL.return_alternative_item(['B', 'C'], s_)
            q_mb = max([q[(s_, a)] for a in self.action_space]) - max([q[(alt_s_, a)] for a in self.action_space])  # left - right

        # scale by beta_mb
        mb_value = self.beta_mb * q_mb

        # softmax choice rule
        p = expit(mb_value)

        # save expecetd value of mb = beta_mb * q_value
        self.markov_state._mb_value = mb_value
        self.markov_state._state1_p = p
        return a

    def evaluate_ibl_mb2(self, response=None):
        """
        Implemented based on to memory sampling idea

        :param response:
        :return:
        """
        assert self.markov_state._curr_stage == '1'
        # Q(MF)
        if not response:
            response = self.action_space[0]  # always eval left

        q = self.q.copy()
        b_value = max([q[('B', a)] for a in self.action_space])
        c_value = max([q[('C', a)] for a in self.action_space])

        # Determine the choice
        # Sampling memory
        estimated_freq = self.sampling_memory(response=response)
        if COMMON_TRANS[response] == 'B':
            q_mb = (2 * estimated_freq - 1) * (b_value - c_value)
        else:
            q_mb = (2 * estimated_freq - 1) * (c_value - b_value)

        # scale by beta_mb
        mb_value = self.beta_mb * q_mb

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == response else -1

        # softmax choice rule
        p = expit(mb_value + rep * self.p_parameter)

        # decide action
        if random.random() < p:
            a = response
        else:
            a = MarkovIBL.return_alternative_item(self.action_space, response)

        # save expected q value of mb = beta_mb * q_value
        self.markov_state._mb_value = mb_value
        self.markov_state._state1_p = p
        return a

    def sampling_memory(self, response, n=10):
        samples = [self.memory.retrieve(rehearse=False, curr_state='A', response=response)['next_state'] for i in range(n)]
        estimated_freq = np.round(samples.count('B')/n, 2)
        return estimated_freq

    def evaluate_ibl_hybrid(self, response=None):
        return self.evaluate_ibl_hybrid1(response=response)

    def evaluate_ibl_hybrid1(self, response=None):
        """
        A combination of RL-MF + IBL-MB
        The potential problem of this model is, no matter what, it will go through MB evaluation process
        :return:
        """
        # value of MB
        mb_response = self.evaluate_ibl_mb(response=response)
        if not response:
            # response = self.markov_state._retrieved_response
            response = mb_response
        mb_value = self.markov_state._mb_value

        # value of MF
        self.evaluate_rl_mf(response=response)
        mf_value = self.markov_state._mf_value

        # calculate blended value
        # hybrid_value = self.w_parameter * mb_value + (1 - self.w_parameter) * mf_value
        hybrid_value = mf_value + mb_value

        # control stickness
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == response else -1

        # softmax choice rule
        p = expit(hybrid_value + rep * self.p_parameter)  # p_left

        if random.random() < p:
            a = response
        else:
            a = MarkovIBL.return_alternative_item(self.action_space, response)

        self.markov_state._state1_p = p
        self.markov_state._hybrid_value = hybrid_value
        return a

    def evaluate_ibl_hybrid2(self, response=None):
        """
        According to Daw 2011
        Evaluate Q table using RL-Hybrid
        >> best parameter combination
            alpha=.5,
            beta=5,
            p_parameter=0,
            lambda_parameter=.6
            beta_mf
            beta_mb
        """
        if not response:
            response = self.action_space[0]  # always eval left

        self.evaluate_rl_mf(response=response)
        self.evaluate_ibl_mb(response=response)

        # note: no need to x beta_mf since mf_value is already multiplied by beta_mf
        hybrid_value = self.markov_state._mf_value + self.markov_state._mb_value

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == response else -1

        # softmax choice rule
        p = expit(hybrid_value + rep * self.p_parameter)  # p_left

        if random.random() < p:
            a = response
        else:
            a = MarkovIBL.return_alternative_item(self.action_space, response)

        # Q(Hybrid)
        self.markov_state._state1_p = p
        self.markov_state._hybrid_value = hybrid_value
        return a



    def retrieve_response_from_state(self, s_):
        """
        Retrieve the response that mostly likely leads to next state s_
        Advance time
        :param s_:
        :return:
        """
        retrieved_memory = self.start_retrieving(rehearse=False, curr_state='A', next_state=s_)
        retrieved_response = retrieved_memory['response']
        self.memory.advance()
        return retrieved_response



    # =================================================== #
    # CHOOSE STATE1
    # =================================================== #

    def choose_rl_mf(self):
        """
        Use model-free algorithm modified based on Feher da Silva 2018
        :return:
        """
        q = self.q.copy()
        if self.markov_state._curr_stage == '1':
            p = self.evaluate_rl_mf()
            if random.random() < p:   # p_left
                a = self.action_space[0]
            else:
                a = self.action_space[1]
            return a
        else:
            # beta: exploration  parameter
            return self.rl_state2_choice()

    def choose_rl_mb(self):
        """
        Use model-base algorithm modified based on Feher da Silva 2018
        :return:
        """
        q = self.q.copy()
        if self.markov_state._curr_stage == '1':
            p = self.evaluate_rl_mb()
            if random.random() < p:
                a = self.action_space[0]
            else:
                a = self.action_space[1]
            return a
        else:
            return self.rl_state2_choice()

    def choose_rl_hybrid(self):
        """
        Choose action using hybrid MF/MB algorithm
        :return:
        """
        q = self.q.copy()
        if self.markov_state._curr_stage == '1':
            p = self.evaluate_rl_hybrid()
            if random.random() < p:
                a = self.action_space[0]
            else:
                a = self.action_space[1]
            return a
        else:
            return self.rl_state2_choice()

    def choose_ibl_mb(self):
        """
        Use IBL algorithm
        :return:
        """
        if self.markov_state._curr_stage == '1':
            a = self.evaluate_ibl_mb()
            return a
        else:
            a = self.ibl_state2_choice()
            return a

    def choose_ibl_hybrid(self):
        """
        Use IBL algorithm
        :return:
        """
        if self.markov_state._curr_stage == '1':
            a = self.evaluate_ibl_hybrid()
            return a
        else:
            a = self.ibl_state2_choice() 
            return a

    # =================================================== #
    # CHOOSE STATE2
    # =================================================== #

    def rl_state2_choice(self):
        assert self.markov_state._curr_stage == '2'
        """Get simulated choice at state2 using RL
        Keyword arguments:
        q: dict of final-state action values
        beta: exploration parameter
        """
        q = self.q.copy()
        probs = np.array(
            [np.exp(self.beta * v) for (s, a), v in q.items() if s == self.markov_state._curr_state])
        probs /= np.sum(probs)
        r = random.random()
        s = 0
        for a, x in enumerate(probs):
            s += x
            if s >= r:
                return self.action_space[a]
        return a

    def ibl_state2_choice(self):
        assert self.markov_state._curr_stage == '2'
        """Get simulated choice at state2 using IBL
        Keyword arguments: 
        """
        assert self.markov_state._curr_stage == '2'
        retrieved_memory = self.start_retrieving(rehearse=False, curr_state=self.markov_state._curr_state)
        if retrieved_memory['reward'] > 0:
            a = retrieved_memory['response']
        else:
            a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])
        return a

    # =================================================== #
    #  RL: UPDATE Q
    # =================================================== #

    def update_q(self):
        """
        Update Q table (RL MF/MB)
        :return:
        """
        q = self.q.copy()
        s = 'A'
        s_ = self.markov_state._curr_state
        a = self.markov_state.state1_response
        a_ = self.markov_state.state2_response
        r = self.markov_state.received_reward

        # print('before', q)
        q[(s, a)] = (1 - self.alpha) * q[(s, a)] + self.alpha * q[(s_, a_)] + \
                 self.alpha * self.lambda_parameter * (r - q[(s_, a_)])
        q[(s_, a_)] = (1 - self.alpha) * q[(s_, a_)] + self.alpha * r
        # print('after', q)
        self.q = q.copy()
        self.markov_state.q = q.copy()

    # =================================================== #
    # IBL: ENCODE MEMORY
    # =================================================== #

    def encode_memory(self):
        """
        Simple encode memory of current trial
        :return:
        """
        s = 'A'
        s_ = self.markov_state._curr_state
        a = self.markov_state.state1_response
        a_ = self.markov_state.state2_response
        r = self.markov_state.received_reward

        # if r > 0:
        self.memory.learn(state='<S%d>' % (1), curr_state=s, next_state=s_, response=a, reward=r)
        self.memory.learn(state='<S%d>' % (2), curr_state=s_, next_state=None, response=a_, reward=r)

        self.memory.advance(self.advance_time)

    def start_blending(self, outcome_attribute, curr_state):
        """
        To better calculate match score, we clear activation history everytime we blend
        :return: blended value
        """
        # clear activation_history
        self.memory.activation_history = []

        # start blending
        bv = self.memory.blend(outcome_attribute, curr_state=curr_state)
        self.memory.advance()

        # calculate match score and retrieval latency
        match_score = MarkovIBL.match_score(self.memory.activation_history)
        retrieve_latency = MarkovIBL.retrieval_time(match_score, fixed_cost=self.fixed_cost, F=self.lf)

        # record latency
        self.markov_state._retrieve_latency = retrieve_latency
        self.record_retrieval_latency(retrieve_latency)

        # reset activation history
        self.memory.activation_history = []
        return bv

    def start_retrieving(self, **kwargs):
        """
        To better calculate activation, we clear activation history everytime we retrieve
        :return:
        """
        # clear activation_history
        self.memory.activation_history = []

        # start retrieving
        retrieved_memory = self.memory.retrieve(**kwargs)
        self.memory.advance()
        # if None, randomly select one chunk from memory
        # satisfying kwargs constraints
        if retrieved_memory is None:
            constraints = {k:kwargs[k] for k in kwargs.keys() if k not in ('rehearse')}
            retrieved_memory = random.choice([d for d in [dict(m) for m in list(self.memory)] if constraints.items() <= d.items()])
            self.memory.activation_history = []
            return retrieved_memory

        # get activation from history records (max activation value of all retrieval candidates)
        activation = MarkovIBL.activation_score(self.memory.activation_history)
        self.markov_state._retrieved_memory_history = random.choice([c for c in self.memory.activation_history if c['name']==retrieved_memory._name])

        # calculate retrieval latency
        retrieve_latency = MarkovIBL.retrieval_time(activation, fixed_cost=self.fixed_cost, F=self.lf)

        # record latency
        self.markov_state._retrieve_latency = retrieve_latency
        self.record_retrieval_latency(retrieve_latency)

        # clear activation_history
        self.memory.activation_history = []

        return retrieved_memory


    def record_retrieval_latency(self, retrieve_latency):
        """
        record the retrieve_latency to markov_state
        :param retrieve_latency:
        :return:
        """

        if self.markov_state._curr_stage == '1':
            self.markov_state.state1_response_time += retrieve_latency
        if self.markov_state._curr_stage == '2':
            self.markov_state.state2_response_time += retrieve_latency



    # =================================================== #
    # ACT-R MATH FUNCTIONS
    # =================================================== #

    @staticmethod
    def retrieval_time(activation, fixed_cost=.585, F=.63):
        """

        :param fixed_cost: perception and encoding
        :param F: :lf parameter in ACT-R, default F=.63, fixed_cost=.585
        :param activation:
        :return: retrieval time
        """
        return fixed_cost + F * np.exp(-activation)

    @staticmethod
    def match_score(activation_history):
        """
        According to ACT-R, a match score, M, is computed as the
        log of the sum over the chunks i in MS of e to the power A(i).  If M
        is greater than or equal to the retrieval threshold of the declarative
        module then the created chunk is placed into the blending buffer with
        a latency computed in the same way the declarative module computes
        retrieval latency using M as the activation of the chunk.
        :param fixed_cost: perception and encoding
        :param F: :lf parameter in ACT-R, default =.63
        :param activation:
        :return: retrieval time
        """
        return np.log(np.sum([np.exp(c['activation']) for c in activation_history]))

    @staticmethod
    def activation_score(activation_history):
        """

        :param activation_history:
        :return:
        """
        return max(c['activation'] for c in activation_history)

    # =================================================== #
    # RUN EXPERIMENT
    # =================================================== #

    def run_experiment(self, n=1):
        """
        """
        for i in range(n):
            self.markov_state = MarkovState()
            self.markov_state.state0()  # init
            self.respond_to_key_press()
            self.respond_to_key_press()
            self.index += 1

    # =================================================== #
    # DATA PROCESSING
    # =================================================== #

    def calculate_stay_probability(self):
        """
        Calculate the probability of stay:
            A trial is marked as "STAY" if the agent selects the same action in current trial (e.g. LEFT)
            as the previous trial

        NOTE: will be -1 trials because exclude NA rows
        """

        df = self.df_behaviors()
        df['state1_stay'] = df['state1_response'].shift()  # first row is NA (look at previsou trial)
        df['state1_stay'] = df.apply(
            lambda x: 1 if x['state1_stay'] == x['state1_response'] else (np.nan if pd.isnull(x['state1_stay']) else 0),
            axis=1)
        # df['pre_received_reward'] = df['received_reward'].shift()
        # df['pre_received_reward'] = df.apply(lambda x: 'non-reward' if x['pre_received_reward'] == 0 else 'reward', axis=1)
        df = df.dropna(subset=['state1_stay', 'pre_received_reward', 'pre_state_frequency'])
        df = df.astype({'pre_state_frequency': CategoricalDtype(categories=['rare', 'common'], ordered=True),
                        'pre_received_reward': CategoricalDtype(categories=['non-reward', 'reward'], ordered=True)})
        df = df[
            ['index', 'state_frequency', 'received_reward', 'pre_received_reward', 'pre_state_frequency', 'state1_stay',
             'state1_response_time', 'state2_response_time']]
        return df

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
        ] for s in self.log]

        df = pd.DataFrame(rows, columns=['state1_response',
                                         'state1_response_time',
                                         'state1_selected_stimulus',
                                         'state2_response',
                                         'state2_response_time',
                                         'state2_selected_stimulus',
                                         'received_reward',
                                         'state_frequency',
                                         'reward_frequency']).reset_index()
        df['pre_received_reward'] = df['received_reward'].shift()
        df['pre_received_reward'] = df.apply(lambda x: x['pre_received_reward'] if pd.isnull(x['pre_received_reward']) \
            else ('non-reward' if x['pre_received_reward'] <= 0 else 'reward'), axis=1)
        df['pre_state_frequency'] = df['state_frequency'].shift()
        return df

    def df_blend(self):
        """
        Prepare df of blended values
        :return:
        """
        rows = [[
            s._best_blended_state,
            s._best_blended_value,
        ] for s in self.log]
        df = pd.DataFrame(rows, columns=['blended_state', 'blended_value'])
        return df

    def df_q_table(self):
        df_q =pd.DataFrame([s.q for s in self.log])
        df_q.columns = [(s, RESPONSE_CODE[a]) for (s, a) in df_q.columns]
        return df_q

    # =================================================== #
    # PARAMETER ESTIMATION
    # =================================================== #
    def estimate_LL(self, df, model_name='markov-rl-mf', init=True, verbose=False, **params):
        """
        trial by trial estimation
        LL += np.log(prob)
        :return:
        """
        # init everything
        if init:
            self.kind = model_name
            self.LL = 0.0
            q = self.q.copy()
            self.q = q.copy().fromkeys(q, 0)
            self.log = []
            self.init_memory()
            self.update_parameters(**params)

        # do not display every trial information
        self.verbose = False

        for i, row in df.iterrows():
            self.markov_state = MarkovState()
            self.markov_state.state0()  # init
            self.respond_from_data(response1=row['state1_response'],
                                   response2=row['state2_response'],
                                   received_reward=row['received_reward'],
                                   state2='B',
                                   state_frequency=row['state_frequency'],
                                   reward_frequency='common')
            self.respond_from_data(response1=row['state1_response'],
                                   response2=row['state2_response'],
                                   received_reward=row['received_reward'],
                                   state2='B',
                                   state_frequency=row['state_frequency'],
                                   reward_frequency='common')
            self.index += 1
        if verbose:
            print('>>> ESTIMATE LOG-LIKELIHOOD %s [SUBJECT: %s] <<<' % (self.kind, df['subject_id'].unique()[0]))
            print('>>> PARAMETERS: %s <<<\n' % str(self.task_parameters))
            print('\t...Log-Likelihood = [%.2f]' % (self.LL))

        return self.LL

    def estimate_mod_coef_(self):
        """
        Fit data into linear model and estimate coef_
        :return:
        """
        df = self.calculate_stay_probability()
        mod = glm(formula="state1_stay ~ pre_state_frequency * pre_received_reward", data=df,
                  family=sm.families.Binomial()).fit()
        dfm = pd.DataFrame(mod.params[1:]).reset_index()
        dfm.columns = ['term', 'coef_']
        self._mod_res = mod
        self._mod_df = dfm
        return dfm

    # =================================================== #
    # HELPER FUNCTIONS
    # =================================================== #

    @staticmethod
    def return_alternative_item(arr, item):
        """
        In any array of two items, return the alternative one
        :param arr:
        :param item:
        :return:
        """
        try:
            return [i for i in arr if i != item][0]
        except:
            print("Error: No %s in" % (item) , arr)
            return None

    def __str__(self):
        header = "################## SETUP MODEL " + self.kind + " ##################\n" + str(self.task_parameters)
        return header

    def __repr__(self):
        return self.__str__()

class MarkovSimulation():
    GROUP_VAR = ['pre_received_reward', 'pre_state_frequency']

    @staticmethod
    def run_single_simulation(model='markov-ibl-mb', n=201, verbose=False, **params):

        m = MarkovIBL(model=model, verbose=verbose)
        m.update_parameters(**params)
        m.memory.activation_history = []
        m.run_experiment(n=n)
        return m

    @staticmethod
    def run_simulations(model='markov-ibl-mb', e=1, n=201, verbose=False, **params):
        df_list = []
        for i in tqdm(range(e)):
            m = MarkovIBL(model=model, verbose=False)
            m.update_parameters(**params)
            if verbose and (not i):
                print(m.__str__())
            m.run_experiment(n=n)
            temp = m.calculate_stay_probability()
            temp['epoch'] = i
            df_list.append(temp)
        res = pd.concat(df_list, axis=0)
        for k,v in params.items():
            if k =='REWARD':
                v = str(set(val for val in v.values()))
            res[k]  = v

        res = res.groupby(['epoch'] +
                          MarkovSimulation.GROUP_VAR).\
            agg(state1_stay_mean=('state1_stay', 'mean'),
              state1_response_time_mean=('state1_response_time', 'mean'),
              state2_response_time_mean=('state2_response_time', 'mean')).reset_index()
        return res

    @staticmethod
    def simulate_param_effect(model_name='markov-ibl-mb', param_name='temperature', e=5, save_output=False, overwrite=False):
        """
        Simulate the effect of various parameter values
        :param model_name:
        :param param_name:
        :param e:
        :param save_output:
        :return:
        """
        # check exist
        if save_output:
            f = os.path.join(save_output, '%s-%s-sim.csv' % (model_name, param_name))
            if os.path.exists(f) and (not overwrite):
                print('...LOAD...')
                dff = pd.read_csv(f)
                return dff

        # start simulation
        r1, r0 = 1, 0
        est = MarkovEstimation(model_name=model_name)
        # params =  dict(zip(est.param_names, est.param_inits))
        params = {'alpha': 0.2,
                  'beta': 2,
                  'beta_mf': 2,
                  'beta_mb': 2,
                  'lambda_parameter': .5,
                  'p_parameter': 0,
                  'temperature': 0.2,
                  'decay': 0.5,
                  'lf': 0.5,
                  'fixed_cost': 0.0}

        lower, upper = dict(zip(est.param_names, est.param_bounds))[param_name]
        param_values = np.linspace(lower, upper, num=8, endpoint=False).round(2)
        df_list = []
        for v in param_values:
            # update parameter
            params[param_name] = v
            df = MarkovSimulation.run_simulations(model=model_name, e=e, verbose=0, **params)
            df[param_name] = v
            df_list.append(df)
        dff = pd.concat(df_list, axis=0)

        # save simulation
        if save_output:
            try:
                f = os.path.join(save_output, '%s-%s-sim.csv' % (model_name, param_name))
                dff.to_csv(f, index=False)
                if overwrite: print("...OVERWRITE...")
            except:
                return dff
        return dff

    @staticmethod
    def simulate_param_recovery(pr_dir, verbose=False):
        """
        Run parameter recovery anlaysis
        :param df_opt:
        :param des_dir:
        :return:
        """
        # define subject and models
        subject_ids = [str(i) for i in np.arange(1, 152)][:2]
        estimate_models = ['markov-rl-hybrid', 'markov-ibl-hybrid']

        d1 = os.path.join(pr_dir, 'opt_original')
        d2 = os.path.join(pr_dir, 'fake_subject')
        d3 = os.path.join(pr_dir, 'opt_recovered')

        if not os.path.exists(d1):
            print('CREATE DIR...')
            os.makedirs(d1, exist_ok=True)
            os.makedirs(d2, exist_ok=True)
            os.makedirs(d3, exist_ok=True)

        # start run original optimization
        if len(os.listdir(d1)) < len(subject_ids)*len(estimate_models):
            if verbose: print('START OPT ORI...')

            # start optimization
            for i in tqdm(range(len(subject_ids))):
                subject_id = subject_ids[i]
                for estimate_model in estimate_models:
                        MarkovEstimation.try_estimate(subject_dir=None, # default
                                                      subject_id=subject_id,
                                                      estimate_model=estimate_model,
                                                      save_output=d1,
                                                      verbose=verbose)

        # start run single simulation
        if len(os.listdir(d2)) < len(subject_ids):
            if verbose: print('START SINGLE RUN SIMULATION...')
            df_opt = MarkovEstimation.load_optimization_data(opt_dir=d1, estimate_models=['markov-ibl-hybrid'])
            for i, row in df_opt.iterrows():
                d = dict(row)
                estimate_model = d['estimate_model']
                subject_id = d['subject_id']
                params = {key: d[key] for key in PARAMETER_NAMES}

                # run single simulation
                m = MarkovSimulation.run_single_simulation(model=estimate_model, verbose=False, **params)
                df_fake = m.df_behaviors()
                df_fake['subject_id']='sub%d' % subject_id

                # save data
                os.makedirs(os.path.join(d2, 'sub%d' % (subject_id)), exist_ok=True)
                df_fake.to_csv(os.path.join(d2, 'sub%d' % (subject_id), 'test.csv'))

        # start run recovered optimization
        if len(os.listdir(d3)) < len(subject_ids)*len(estimate_models):
            if verbose: print('START OPT REC...')

            # start optimization
            for i in tqdm(range(len(subject_ids))):
                subject_id = subject_ids[i]
                for estimate_model in estimate_models:
                    MarkovEstimation.try_estimate(subject_dir=d2,
                                                  subject_id=subject_id,
                                                  estimate_model=estimate_model,
                                                  save_output=d3,
                                                  verbose=verbose)


class MarkovEstimation():
    def __init__(self, model_name='markov-rl-mf', subject_dir=None, subject_id=None, drop_first_9=False, verbose=False):
        self.kind = model_name
        self.verbose = verbose

        # init parameter bounds, init values and load subject data
        self.init_estimated_params()
        self.init_data(subject_dir=subject_dir, subject_id=subject_id, drop_first_9=drop_first_9)

    def init_estimated_params(self):
        """
        As in Decker et al. (2016) and Potter, Bryce et al. (2017)
        TODO: implement estimation for response time parameter lf, fixed costs
        :return:
        """
        param_names = ['alpha', 'beta', 'beta_mf', 'beta_mb', 'lambda_parameter', 'p_parameter', 'temperature', 'decay', 'lf', 'fixed_cost']
        # param_bounds = {'alpha':(0,1), 'beta':(0,30), 'beta_mf':(0,30), 'beta_mb':(0, 30), 'lambda_parameter':(0,1), 'p_parameter':(-30,30), 'temperature':(0.01,1), 'decay':(0.01,1.6), 'lf':(0.01,1), 'fixed_cost':(0,1)}
        param_bounds = {'alpha':(0,1), 'beta':(0,5), 'beta_mf':(0,5), 'beta_mb':(0,5), 'lambda_parameter':(0,1), 'p_parameter':(-2,2), 'temperature':(0.01,2), 'decay':(0.01,1.5), 'lf':(0.01,1), 'fixed_cost':(0,1)}
        param_zero_bounds = {'alpha':(0.5,0.5), 'beta':(2,2), 'beta_mf':(2,2), 'beta_mb':(2,2), 'lambda_parameter':(0.5,0.5), 'p_parameter':(.5,.5), 'temperature':(0.2,0.2), 'decay':(0.5,0.5), 'lf':(0.5,0.5), 'fixed_cost':(0,0)}

        if self.kind == 'markov-rl-mf':
            exclude = ['beta_mb', 'temperature', 'decay', 'lf', 'fixed_cost']

        elif self.kind == 'markov-rl-mb':
            exclude = ['beta_mf', 'temperature', 'decay', 'lf', 'fixed_cost']

        elif self.kind == 'markov-rl-hybrid':
            exclude = ['temperature', 'decay', 'lf', 'fixed_cost'] # exclude latency parameter

        elif self.kind == 'markov-ibl-mb':
            exclude = ['alpha', 'beta', 'beta_mf', 'beta_mb', 'lambda_parameter', 'p_parameter', 'lf', 'fixed_cost'] # exclude latency parameter
            exclude = ['beta', 'lf', 'fixed_cost'] # exclude latency parameter
        elif self.kind == 'markov-ibl-hybrid':
            exclude = ['beta', 'lf', 'fixed_cost'] # exclude latency parameter
        else:
            pass

        bounds = param_bounds.copy()
        bounds.update({key: param_zero_bounds[key] for key in exclude})
        self.param_names = param_names
        self.param_bounds = [v for k, v in bounds.items() if k in self.param_names]
        self.param_inits = [np.round(np.random.uniform(l, u), 2) for (l, u) in self.param_bounds]

        # grid search
        self.param_ls = [np.unique(np.round(np.linspace(l, u, num=5), 2)) for (l, u) in self.param_bounds]
        self.param_gs_ls = [dict(zip(self.param_names, c))for c in list(itertools.product(*self.param_ls))]

        # self.param_init = [np.random.beta(1.1,1.1), # alpha
        #                    np.random.gamma(3, 1),   # beta
        #                    np.random.gamma(3, 1),   # beta_mf
        #                    np.random.gamma(3, 1),   # beta_mb
        #                    np.random.beta(1.1,1.1), # lambda_parameter
        #                    np.random.normal(0,10),  # p_parameter
        #                    np.random.normal(0, 1),  # temperature
        #                    np.random.normal(0, 1)]  # decay

    def init_data(self, subject_dir, subject_id, drop_first_9):
        """
        According to Decker et al. (2016) and Potter, Bryce et al. (2017)
        drop first 9 trials when estimate
        :param data:
        :param drop_first_10:
        :return:
        """
        if not subject_dir:
            self.subject_dir = os.path.join(os.path.dirname(os.getcwd()), 'data', 'human', 'online_data')
        else:
            self.subject_dir = subject_dir

        if not subject_id:
            self.data = None
            return
        else:
            try:
                self.data = MarkovEstimation.load_subject_data(subject_dir=self.subject_dir, subject_id=subject_id)
            except:
                print('Cannot find subject data...Check data path', self.subject_dir)

        if drop_first_9:
            self.data = self.data.iloc[9:,]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @staticmethod
    def load_subject_data(subject_dir, subject_id=None):
        df = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(subject_dir, '*', '', 'test.csv'))], axis=0)

        # choice processing
        # state1_response = 1 (49, f), 2 (48, k)
        df['state1_response'] = df['state1_response'].replace({1: 'f', 2: 'k'})
        df['state2_response'] = df['state2_response'].replace({1: 'f', 2: 'k'})
        try:
            return df[df['subject_id']=='sub%s' % (subject_id)]
        except:
            return df

    @staticmethod
    def load_opt_parameters(opt_dir, subject_id, estimate_model, only_maxLL=True, verbose=False):
        """
        Load optimized parameter data
        :param opt_dir: os.path.join(main_dir, 'data', 'model', 'param_optimization')
        :param subject_id: os.path.join(main_dir, 'data', 'human', 'online_data')
        :param estimate_model: 'markov-rl-mf'
        :return:
            d {'alpha': 0.1674,
                 'beta': 9.9208,
                 'lambda_parameter': 0.7553,
                 'p_parameter': 0.051,
                 'w_parameter': 0.0,
                 'temperature': 0.1,
                 'decay': 0.1,
                 'maxLL': -9.6775,
                 'estimate_model': 'markov-rl-mf',
                 'subject_id': 'sub1'}
            params: {'alpha': 0.1674,
                 'beta': 9.9208,
                 'lambda_parameter': 0.7553,
                 'p_parameter': 0.051,
                 'w_parameter': 0.0}
        """

        f = os.path.join(opt_dir, 'sub%s-%s-opt-result.csv' % (subject_id, estimate_model))
        if not os.path.exists(f):
            if verbose: print('Cannot find file')
            return
        df = pd.read_csv(f)
        if only_maxLL:
            d = df.loc[df['maxLL'].idxmax()].to_dict()
            params = dict([(k, v) for k, v in d.items() if k in RL_PARAMETER_NAMES])
            return d, params
        else:
            return df

    @staticmethod
    def load_optimization_data(opt_dir, estimate_models=None, long_format=False, only_maxLL=False):
        """
        load optimization data into df
        :param opt_dir:
        :param estimate_models:
        :param only_maxLL:
        :return: dataframe
        """
        if estimate_models is None:
            estimate_models = ['markov-rl-mf',
                               'markov-rl-mb',
                               'markov-rl-hybrid',
                               'markov-ibl-mb',
                               'markov-ibl-hybrid']
        ls = []
        for i in np.arange(1, 152):
            for estimate_model in estimate_models:
                try:
                    if only_maxLL:
                        d, _ = MarkovEstimation.load_opt_parameters(opt_dir=opt_dir, subject_id=str(i),
                                                                    estimate_model=estimate_model, only_maxLL=True)
                        df = pd.Series(d)
                    else:
                        df = MarkovEstimation.load_opt_parameters(opt_dir=opt_dir, subject_id=str(i),
                                                                  estimate_model=estimate_model, only_maxLL=False)
                    ls.append(df)
                except:
                    print('Unable to find SUB - %s' % i)
        if only_maxLL:
            df = pd.DataFrame(ls)
        else:
            df = pd.concat(ls, axis=0)
        if long_format:
            df = df.melt(id_vars=['subject_id', 'estimate_model'], var_name='param_name', value_name='param_value')
        return df

    @staticmethod
    def boltzmann(options, values, temperature):
        """Returns a Boltzmann distribution of the probabilities of each option"""
        temperature = max(temperature, 0.01)
        vals = np.array(values) / temperature
        bvals = np.exp(vals) / np.sum(np.exp(vals))
        return dict(zip(options, bvals))

    def estimate_function(self, param_values = None):
        """
        Estimate LL for instance of
        :param model_name:
        :param verbose:
        :param args: a list of values
        :return:
        """
        assert (self.data is not None) and (len(param_values) == 8 | len(param_values) == 10)
        # define parameters
        param_dict = dict(zip(PARAMETER_NAMES, param_values))
        est_instance = MarkovIBL(verbose=False)

        LL = est_instance.estimate_LL(df=self.data, init=True, verbose=self.verbose, model_name=self.kind,  **param_dict)
        # LL = MarkovEstimation.estimate_postLL(df=self.data, model_name=self.kind, e=50, verbose=self.verbose, **param_dict)
        return LL

    def v_function(self, param_values):
        """
        Vector function for optimization
        Multiply by -1 to minimize = maximize LL
        :param args:
        :return:
        """
        assert (self.data is not None) and (len(param_values) > 0)
        # alpha = x0[0]
        # beta = x0[1]
        # ...
        return -1 * self.estimate_function(param_values=param_values)

    @staticmethod
    def optimization_function(df, x0, param_bounds, estimate_model='markov-rl-mf', drop_first_9=False):
        """
        Optimization
        :param df:
        :param x0: ['alpha', 'beta', 'lambda_parameter', 'p_parameter', 'w_parameter', 'temperature', 'decay']
        :param param_bounds:
        :return:

         >> res = MarkovEstimateion.optimization_function(df=df, x0=init_params, param_bounds=param_bounds)
        """
        # create an estimation instance
        # define estimate model name and pass in data
        # est = MarkovEstimation(data=df, model_name=estimate_model)estimate_LL
        est = MarkovEstimation(subject_id=None, model_name=estimate_model, drop_first_9=drop_first_9, verbose=False)
        est.data = df

        # start optimization
        res = opt.minimize(est.v_function, x0=x0, bounds=param_bounds, method="Nelder-Mead")
        return res

    @staticmethod
    def try_estimate(subject_dir=None, subject_id='1', estimate_model='markov-rl-mf', save_output=False, verbose=False):
        """
        Try to estimate maxLL of a subject with a specific model
        According to Decker 2016,
            - drop first 9 trials
            - only estimate 8 parameters (exclude latency parameters: lf and fixed cost)
            - applied following priors and bounds to parameters
            - randomly initialized parameter values
            - run 10 times
        :param subject_dir: if None, use default subject_dir
        :param subject_id:
        :param estimate_model:
        :return: a dataframe of all model MaxLL
        """
        est = MarkovEstimation(subject_dir=subject_dir, model_name=estimate_model, subject_id=subject_id, verbose=verbose)
        res = MarkovEstimation.optimization_function(df=est.data, x0=est.param_inits, param_bounds=est.param_bounds)

        dfp = pd.DataFrame(
            {**dict(zip(est.param_names, res['x'])),
             'maxLL':-1*res['fun'],
             'estimate_model': estimate_model,
             'subject_id': subject_id},
            index=[0]).round(4)

        if not save_output:
            return dfp

        # define dest path
        dest_file = os.path.join(save_output, 'sub%s-%s-opt-result.csv' % (subject_id, estimate_model))

        # append optimization if exist
        if os.path.exists(dest_file):
            mode = 'a'
            header = False
        else:
            mode = 'w'
            header = True
        dfp.to_csv(dest_file, index=False, mode=mode, header=header)
        if verbose: print('... SAVED optimized parameter data ...[%s] [%s]' %(subject_id, estimate_model))
        return dfp

    @staticmethod
    def estimate_postLL(df, model_name='markov-rl-mf', e=100, verbose=False, **params):
        """

        :param df: subject data
        :param model_name: estimated model name
        :param init: init parameter
        :param verbose:
        :param params:
        :return:
        """
        df_model = MarkovSimulation.run_simulations(model=model_name, e=e, verbose=verbose, **params)
        df_subject = df
        LL = MarkovEstimation.calculate_LL(df_model=df_model, df_subject=df_subject,
                                           factor_cols=['pre_received_reward', 'pre_state_frequency'],
                                           dv_name='state1_stay', return_numeric=True)
        return LL

    @staticmethod
    def calculate_LL(df_model, df_subject, factor_cols=['pre_received_reward', 'pre_state_frequency'],
                     dv_name='state1_stay', return_numeric=True):
        """
        Calcualte LL of PSWitch
        df_merge: df_model + df_subject merged dataframe
        Since df_merge contains [ans.m, bll.m]
        """
        # merge dataframes
        df_model = df_model.groupby(factor_cols).agg(state1_stay_mean=(dv_name + '_mean', 'mean'), state1_stay_std=(dv_name + '_mean', 'std')).reset_index()
        df_subject = df_subject.groupby(['subject_id'] + factor_cols).agg(state1_stay_mean=(dv_name, 'mean'), state1_stay_std=(dv_name, 'std')).reset_index()
        df_merge = pd.merge(df_subject, df_model, how='outer', on=factor_cols, suffixes=('.s', '.m'))

        # do ca;ci;ayiopm
        df_merge[dv_name + '_z'] = df_merge.apply(
            lambda x: (x[dv_name + '_mean.m'] - x[dv_name + '_mean.s']) / max(x[dv_name + '_std.m'], 1e-10), axis=1)
        df_merge[dv_name + '_probz'] = df_merge.apply(lambda x: norm.pdf(x[dv_name + '_z']), axis=1)
        df_merge[dv_name + '_logprobz'] = df_merge.apply(lambda x: np.log(max(x[dv_name + '_probz'], 1e-10)), axis=1)

        # df_LL = df_merge.agg(LL=('state1_stay_logprobz', 'sum')).reset_index()
        LL = df_merge[dv_name + '_logprobz'].sum()
        if return_numeric:
            return LL
        df_merge['LL'] = LL
        return df_merge

    @staticmethod
    def try_estimate_grid_search(dest_dir, model_name='markov-rl-hybrid', verbose=False):
        """
        Grid search optimization
        :param dest_dir:
        :param model_name:
        :param verbose:
        :return:
        """

        dest_dir = os.path.join(dest_dir, model_name)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
            if verbose: print('...CREATE ', dest_dir)

        est = MarkovEstimation(model_name=model_name)

        # prepare param log
        dfp = pd.DataFrame(est.param_gs_ls)
        dfp['model_name'] = est.kind
        dfp['param_id'] = ['param_id%05d'% i for i in dfp.index]
        dfp.to_csv(os.path.join(dest_dir, '%s-param-log.csv' %(model_name)), index=False)

        # start grid search simulation
        param_id = 0
        for params in est.param_gs_ls:
            df = MarkovSimulation.run_simulations(model=model_name, e=100, verbose=verbose, **params)
            df['param_id'] = param_id
            df['model_name'] = est.kind

            dest_f = os.path.join(dest_dir, '%s-param_id%05d-sim.csv' % (model_name, param_id))
            df.to_csv(dest_f, index=False)
            param_id += 1
            if verbose: print('...SAVE [%s] [%d]' % (est.kind, param_id))

class MarkovPlot(Plot):
    """
    Inherit from Pot
    """
    @staticmethod
    def plot_param_effect(df, model_name, param_name, combination=False):
        if combination:
            MarkovPlot.plot_param_effect_comb(df, model_name, param_name)
            return
        g = sns.FacetGrid(df, col=param_name, col_wrap=4)
        g.map_dataframe(sns.pointplot, x=Plot.REWARD_FACTOR, y='state1_stay_mean',
                        hue=Plot.TRANS_FACTOR, errorbar='se',
                        palette=Plot.PALETTE,
                        order=['reward', 'non-reward'],
                        hue_order=['common', 'rare'])
        g.add_legend()
        g.refline(y=.5)
        g.tight_layout()
        g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
        g.fig.suptitle('PStay Effect of parameter [%s] on [%s]' % (param_name, model_name))

    @staticmethod
    def plot_param_effect_comb(df, model_name, param_name):
        fig, ax = plt.subplots(figsize=(Plot.FIG_WIDTH, Plot.FIT_HEIGHT))
        fig.suptitle('Summary: PStay Effect of [%s] on [%s]' % (param_name, model_name))
        ax = sns.pointplot(data=df[df[Plot.TRANS_FACTOR] == 'common'],
                           x=Plot.REWARD_FACTOR, y='state1_stay_mean',
                           hue=param_name,
                           palette='Blues',
                           order=['reward', 'non-reward'])
        ax = sns.pointplot(data=df[df[Plot.TRANS_FACTOR] == 'rare'],
                           x=Plot.REWARD_FACTOR, y='state1_stay_mean',
                           hue=param_name,
                           palette='Reds',
                           order=['reward', 'non-reward'])
        ax.axhline(0.5, color='grey', ls='-.', linewidth=.5)
        sns.move_legend(ax, "upper right", ncol=2, title=param_name, frameon=False)
        plt.tight_layout()

    @staticmethod
    def plot_param_effect_rt(df, model_name, param_name, combination=False):
        df['response_time_mean'] = df.apply(lambda x: x['state1_response_time_mean'] + x['state2_response_time_mean'],
                                            axis=1)
        if combination:
            MarkovPlot.plot_param_effect_rt_comb(df, model_name, param_name)
            return
        g = sns.FacetGrid(df, col=param_name, col_wrap=3)
        g.map_dataframe(sns.barplot, x=Plot.TRANS_FACTOR, y='response_time_mean',
                        errorbar='se',
                        palette=Plot.PALETTE,
                        order=['common', 'rare'])
        g.map_dataframe(sns.pointplot, x=Plot.TRANS_FACTOR, y='response_time_mean',
                        errorbar='sd', color='black',
                        order=['common', 'rare'])
        g.add_legend()
        g.refline(y=1.09)
        g.tight_layout()
        g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
        g.fig.suptitle('RT Effect of parameter [%s] on [%s]' % (param_name, model_name))

    @staticmethod
    def plot_param_effect_rt_comb(df, model_name, param_name):
        fig, ax = plt.subplots(figsize=(Plot.FIG_WIDTH, Plot.FIT_HEIGHT))
        fig.suptitle('Summary: RT Effect of [%s] on [%s]' % (param_name, model_name))
        ax = sns.pointplot(data=df, x=Plot.TRANS_FACTOR, y='response_time_mean',
                           hue=param_name,
                           palette='Greys',
                           order=['common', 'rare'])
        plt.tight_layout()
        # fig, ax = plt.subplots(figsize=(Plot.FIG_WIDTH, Plot.FIT_HEIGHT))
        # fig.suptitle('Summary: RT Effect of [%s] on [%s]' % (param_name, model_name))
        # ax = sns.pointplot(data=df[df[Plot.TRANS_FACTOR] == 'common'],
        #                    x=Plot.REWARD_FACTOR, y='response_time_mean',
        #                    hue=param_name, #dodge=.1, se='se',
        #                    palette='Blues',
        #                    order=['reward', 'non-reward'])
        # ax = sns.pointplot(data=df[df[Plot.TRANS_FACTOR] == 'rare'],
        #                    x=Plot.REWARD_FACTOR, y='response_time_mean',
        #                    hue=param_name, #dodge=.2,
        #                    palette='Reds',
        #                    order=['reward', 'non-reward'])
        # ax.axhline(1.09, color='grey', ls='-.', linewidth=.5)
        # plt.tight_layout()


    @staticmethod
    def plot_response_switch(df, model_name, dep_var_suffix='', barplot=True):
        """
        Plot state1_stay by pre_received_reward and pre_state_frequency
        :param df:
        :param model_name:
        :return:
        """
        assert set(Plot.PLOT_FACTOR_VAR + ['state1_stay' + dep_var_suffix]).issubset(set(df.columns))
        if len(dep_var_suffix) > 0:
            se = 'se'  # enable se
        else:
            se = None

        fig, ax = plt.subplots(figsize=(Plot.FIG_WIDTH, Plot.FIT_HEIGHT))
        fig.suptitle('Summary: Stay Probability \n[%s]' % (model_name))
        # pointplot
        if not barplot:
            sns.pointplot(data=df, x=Plot.REWARD_FACTOR, y='state1_stay' + dep_var_suffix,
                          hue=Plot.TRANS_FACTOR, errorbar=se,
                          palette=Plot.PALETTE, dodge=True,
                          order=['reward', 'non-reward'],
                          hue_order=['common', 'rare'],
                          ax=ax)
        # barplot
        else:
            sns.barplot(data=df, x=Plot.REWARD_FACTOR, y='state1_stay' + dep_var_suffix,
                        hue=Plot.TRANS_FACTOR, errorbar=se,
                        palette=Plot.PALETTE, alpha=.8,
                        order=['reward', 'non-reward'],
                        hue_order=['common', 'rare'],
                        ax=ax)

            # for container in ax.containers:
            #     ax.bar_label(container, fmt='%.2f', label_type='center')
        ax.axhline(0.5, color='grey', ls='-.', linewidth=.5)
        ax.set_ylim(0, 1.1)
        ax.legend().remove()
        plt.show()

    @staticmethod
    def parameter_lm_plot(df, x_name, y_name, exclude_parameters=None, alpha=.1):
        if exclude_parameters:
            df = df[~df['param_name'].isin(exclude_parameters)]
        g = sns.lmplot(data=df, x=x_name, y=y_name,
                       col="param_name", col_wrap=3, hue="param_name",
                       height=3, aspect=1.2, palette='Set1',
                       scatter_kws={'alpha': alpha},
                       facet_kws=dict(sharex=False, sharey=False))
        g.map_dataframe(MarkovPlot.annotate, x=x_name, y=y_name)
        g.fig.subplots_adjust(top=.9)  # adjust the Figure in rp
        g.fig.suptitle('Correlation of optimizated parmeters [%s] vs. [%s]: ' % (x_name, y_name))
        plt.show()

    @staticmethod
    def annotate(data, **kws):
        x, y = kws['x'], kws['y']
        r, p = stats.pearsonr(data[x], data[y])
        ax = plt.gca()
        minx, miny, maxx, maxy = data[x].min(), data[y].min(), data[x].max(), data[y].max()
        ax.text(0.05, 0.05, 'r = %.2f (p=%.2g, %s)' % (r, p, MarkovPlot.sig(p)), transform=ax.transAxes)
        # ax.text(.05, 0.7, 'x: [%.2f - %.2f], y: [%.2f - %.2f]' % (min1, max1, min2, max2),
        #         transform=ax.transAxes)

    @staticmethod
    def sig(p):
        sig = ''
        if p > .05:
            sig = 'ns'
        if p <= .05:
            sig = '*'
        if p < .01:
            sig = '**'
        if p < .001:
            sig = '***'
        return sig

