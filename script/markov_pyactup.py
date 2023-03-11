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
from tqdm.auto import tqdm
import glob

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
PARAMETER_NAMES = ['beta', 'beta1', 'beta2', 'alpha', 'alpha1', 'alpha2', 'lambda', 'p', 'w', 'noise', 'decay']

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
        self.advance_time = 40

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

        # RL parameters ['beta', 'beta1', 'beta2', 'alpha', 'alpha1', 'alpha2', 'lambda', 'p', 'w', 'noise', 'decay']
        # RL MF/MB parameters
        self.alpha = .2  # learning rate
        # self.alpha1 = .2  # learning rate
        # self.alpha2 = .2  # learning rate
        self.beta = 5  # exploration rate
        # self.beta1 = 5  # exploration rate
        # self.beta2 = 5  # exploration rate

        self.noise = .2  # noise parameter
        self.decay = .2  # decay parameter
        self.lambda_parameter = .2  # decay rate
        self.p_parameter = 0  # perseveration rate
        self.w_parameter = 0  # w = 0: pure MF

        # IBL parameters
        self.memory.noise = self.noise
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
                                'beta': self.beta,  # learning rate
                                # 'beta1' : self.beta1,  # exploration rate
                                # 'beta2' : self.beta2,  # exploration rate
                                'lambda_parameter' : self.lambda_parameter,  # decay rate
                                'p_parameter' : self.p_parameter,  # decay rate
                                'w_parameter' : self.w_parameter,  # w = 0: pure MF
                                # IBL Parameter
                                'noise' : self.noise,  # noise rate
                                'decay' : self.decay  # decay rate
        }

    def update_parameters(self, **kwargs):
        """
        Update parameter
        :param kwargs:
        :return:
        """
        self.task_parameters.update(**kwargs)

        # update RL parameters
        self.alpha = self.task_parameters['alpha']
        # self.alpha1 = self.task_parameters['alpha1']
        # self.alpha2 = self.task_parameters['alpha2']
        self.beta = self.task_parameters['beta']
        # self.beta1 = self.task_parameters['beta1']
        # self.beta2 = self.task_parameters['beta2']
        self.noise = self.task_parameters['noise']
        self.decay = self.task_parameters['decay']
        self.lambda_parameter = self.task_parameters['lambda_parameter']
        self.p_parameter = self.task_parameters['p_parameter']
        self.w_parameter = self.task_parameters['w_parameter']

        # update IBL parameters
        self.memory.noise = self.noise
        self.memory.decay = self.decay

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
        key = None
        if self.kind == 'markov-ibl-mb':
            key = self.choose_ibl_mb()
        elif self.kind == 'markov-ibl-hybrid':
            key = self.choose_ibl_hybrid()
        elif self.kind =='markov-rl-mf':
            key = self.choose_rlmf()
        elif self.kind =='markov-rl-mb':
            key = self.choose_rlmb()
        # simple mf/mb hybrid:
        elif self.kind =='markov-rl-hybrid':
            key = self.choose_rlhybrid()
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
            self.markov_state.state1_response_time = 0.0


        else:
            self.markov_state.state2(response)
            self.markov_state.state2_response_time = 0.0

            # continue deliver rewards, no need to wait for response
            self.update_random_walk_reward_probabilities()
            self.markov_state.reward()

        # log
        if self.markov_state.state == 3:
            # update q
            if self.kind == 'markov-ibl-mb':
                # self.encode_memory() # this will wipe out the difference between common/rare
                self.encode_memory_prediction_error()          # not pure MB some degree of hybrid (the best so far, slow)

            elif self.kind.startswith('markov-ibl-hybrid'):
                self.encode_memory() # this will wipe out the difference between common/rare
                # self.encode_memory_alternative()             # some degree of hybrid
                # self.encode_memory_alternative_frequency()

                # create a strange biased pattern
                # self.encode_memory_reward()
                # self.encode_memory_punishment()
            elif self.kind.startswith('markov-rl'):
                self.update_q()
            else:
                pass

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
    # EVALUATE STATE2
    # =================================================== #

    def evaluate_ibl_deprecated(self):
        """
        Complicated evaluate (deprecated for now)
        Not fully undestand...
        :return:
        """
        # b_value, c_value = [np.round(self.memory.blend("reward", curr_state=state), 4) for state in ['B', 'C']]
        # best_val = np.max([b_value, c_value])

        # decide best option
        # state_blend_dict = dict(zip(['B', 'C'], [b_value, c_value]))
        # best_blended_state = random.choice([k for k, v in state_blend_dict.items() if v == best_val])

        best_blended_state, best_blended_val = self.memory.best_blend("reward", ({"curr_state": state} for state in ("B", "C")))
        best_blended_state = best_blended_state['curr_state']
        retrieved_memory = self.memory.retrieve(rehearse=False, curr_state='A', next_state=best_blended_state)

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == self.action_space[1] else -1

        # softmax choice rule
        p = expit(self.beta * (best_blended_val + rep * self.p_parameter))
        # print('best_blended_val = %.2f, rep, %d, p = %.2f' % (best_blended_val, rep, p))
        # if retrieved_memory['reward'] > 0:
        #     a = retrieved_memory['response']
        # else:
        #     a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])

        # if (retrieved_memory['reward'] > 0) and (p < random.random()):
        #     a = retrieved_memory['response']
        # else:
        #     a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])

        a1 = retrieved_memory['response']
        a2 = random.choice([action for action in self.action_space if action != retrieved_memory['response']])
        rand_p = random.random()
        # this commented block show hybrid pattern
        # if random.random() < p:
        #     if retrieved_memory['reward'] > 0:
        #         a = a1
        #     else:
        #         a = a2
        # else:
        #     if retrieved_memory['reward'] > 0:
        #         a = a2
        #     else:
        #         a = a1

        if retrieved_memory['reward'] > 0:
            if rand_p < p:
                a = a1
            else:
                a = a2
        else:
            if rand_p < p:
                a = a1
            else:
                a = a2
        # print('blend = %.2f, rep, %d, p = %.2f, a1[%s] a[%s]' % (best_blended_val, rep, p, a1, a))

        self.memory.advance()
        self.markov_state._best_blended_state = best_blended_state
        self.markov_state._best_blended_value = best_blended_val
        return a

    def evaluate_rlmf(self):
        """
        Evaluate Q table using RL-MF
        >> best parameter combination
            alpha=.5,
            beta=5,
            p_parameter=0,
            lambda_parameter=.6
        :return:
        """
        assert self.markov_state._curr_stage == '1'
        q = self.q.copy()

        # Q(MF)
        q_mf = q[('A', self.action_space[0])] - q[('A', self.action_space[1])]

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == self.action_space[0] else -1

        # softmax choice rule
        # p = expit(self.beta * (q[('A', self.action_space[0])] - q[('A', self.action_space[1])] + rep * self.p_parameter))
        p = expit(self.beta * (q_mf + rep * self.p_parameter))

        self.markov_state._q_mf = q_mf
        self.markov_state._state1_p = p
        return p

    def evaluate_rlmb(self):
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
        q = self.q.copy()
        b_value = max([q[('B', a)] for a in self.action_space])
        c_value = max([q[('C', a)] for a in self.action_space])

        # Determine the choice
        if COMMON_TRANS[self.action_space[0]] == 'B':
            q_mb = (2 * .7 - 1) * (b_value - c_value)
        else:
            q_mb = (2 * .7 - 1) * (c_value - b_value)

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == self.action_space[1] else -1

        # softmax choice rule
        p = expit(self.beta * (q_mb + rep * self.p_parameter)) # p_right

        # Q(MB)
        self.markov_state._state1_p = p
        self.markov_state._q_mb = q_mb
        return p

    def evaluate_rlhybrid(self):
        """
        According to Daw 2011
        Evaluate Q table using RL-Hybrid
        >> best parameter combination
            alpha=.5,
            beta=5,
            p_parameter=0,
            lambda_parameter=.6
            w_parameter=.5
        w_parameter=1 --> MB
        w_parameter=0 --> MF
        :return:
        """
        self.evaluate_rlmf()
        self.evaluate_rlmb()
        q_hybrid = self.w_parameter * self.markov_state._q_mb + (1 - self.w_parameter) * self.markov_state._q_mf

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == self.action_space[1] else -1

        # softmax choice rule
        p = expit(self.beta * (q_hybrid + rep * self.p_parameter))  # p_right

        # Q(Hybrid)
        self.markov_state._state1_p = p
        self.markov_state._q_hybrid = q_hybrid
        return p

    def evaluate_ibl_hybrid(self):
        """
        IBL: MB
        :return:
        ################## SETUP MODEL markov-ibl ##################
        {'MARKOV_PROBABILITY': 0.7, 'REWARD_PROBABILITY': 'LOAD',
        'REWARD': {'B1': (1, -1), 'B2': (1, -1), 'C1': (1, -1), 'C2': (1, -1)},
        'alpha1': 0.5, 'alpha2': 0.5, 'beta1': 2, 'beta2': 5, 'lambda_parameter': 0.5, 'p_parameter': 0,
        'w_parameter': 0, 'noise': 0.2, 'decay': 0.5}
        """
        best_blended_state, best_blended_val = self.memory.best_blend("reward", ({"curr_state": state} for state in ("B", "C")))
        best_blended_state = best_blended_state['curr_state']
        retrieved_memory = self.memory.retrieve(rehearse=False, curr_state='A', next_state=best_blended_state)

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == self.action_space[1] else -1

        # softmax choice rule
        p = expit(self.beta * (best_blended_val + rep * self.p_parameter))

        if random.random() < p:
            a = retrieved_memory['response']
        else:
            a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])

        self.memory.advance()
        self.markov_state._best_blended_state = best_blended_state
        self.markov_state._best_blended_value = best_blended_val
        return a

    def evaluate_ibl_mb(self):
        """
        IBL: two blended value version
        :return:
        ################## SETUP MODEL markov-ibl ##################
        {'MARKOV_PROBABILITY': 0.7, 'REWARD_PROBABILITY': 'LOAD',
        'REWARD': {'B1': (1, -1), 'B2': (1, -1), 'C1': (1, -1), 'C2': (1, -1)},
        'alpha1': 0.5, 'alpha2': 0.5, 'beta1': 5, 'beta2': 5, 'lambda_parameter': 0.2, 'p_parameter': 0,
        'w_parameter': 0, 'noise': 0.1, 'decay': 0.5}
        """
        # blend B and C
        b_value, c_value = [self.memory.blend("reward", curr_state=state) for state in ['B', 'C']]
        self.markov_state._b_value = b_value
        self.markov_state._c_value = c_value
        best_blended_val = np.max([b_value, c_value])

        # decide best_blended_state
        d = dict(zip(['B', 'C'], [b_value, c_value]))
        best_blended_state = random.choice([k for k, v in d.items() if v == best_blended_val])

        # retrieve response
        retrieved_memory = self.memory.retrieve(rehearse=False, curr_state='A', next_state=best_blended_state)

        # rep(a)
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)
        rep = 1 if prev_choice == self.action_space[1] else -1

        # softmax choice rule
        p = expit(self.beta * (best_blended_val + rep * self.p_parameter))

        if random.random() < p:
            a = retrieved_memory['response']
        else:
            a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])

        self.memory.advance()
        self.markov_state._best_blended_state = best_blended_state
        self.markov_state._best_blended_value = best_blended_val
        return a

    # =================================================== #
    # CHOOSE STATE1
    # =================================================== #

    def choose_rlmf(self):
        """
        Use model-free algorithm modified based on Feher da Silva 2018
        :return:
        """
        q = self.q.copy()
        if self.markov_state._curr_stage == '1':
            p = self.evaluate_rlmf()
            if random.random() < p:   # p_left
                a = self.action_space[0]
            else:
                a = self.action_space[1]
            return a
        else:
            # beta: exploration  parameter
            return self.rl_state2_choice()

    def choose_rlmb(self):
        """
        Use model-base algorithm modified based on Feher da Silva 2018
        :return:
        """
        q = self.q.copy()
        if self.markov_state._curr_stage == '1':
            p = self.evaluate_rlmb()
            if random.random() < p:
                a = self.action_space[0]
            else:
                a = self.action_space[1]
            return a
        else:
            return self.rl_state2_choice()

    def choose_rlhybrid(self):
        """
        Choose action using hybrid MF/MB algorithm
        :return:
        """
        q = self.q.copy()
        if self.markov_state._curr_stage == '1':
            p = self.evaluate_rlhybrid()
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
        retrieved_memory = self.memory.retrieve(rehearse=False, curr_state=self.markov_state._curr_state)
        if retrieved_memory['reward'] > 0:
            a = retrieved_memory['response']
        else:
            a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])
        self.memory.advance()
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

    def encode_memory_reward(self):
        """
        This will create a biased memory encoding: sensitive only to
        :return:
        """
        s = 'A'
        s_ = self.markov_state._curr_state
        a = self.markov_state.state1_response
        a_ = self.markov_state.state2_response
        r = self.markov_state.received_reward

        if r > 0:
            self.memory.learn(state='<S%d>' % (1), curr_state=s, next_state=s_, response=a, reward=r)
            self.memory.learn(state='<S%d>' % (2), curr_state=s_, next_state=None, response=a_, reward=r)
            # encode alternative state will make hybrid become MB
            alt_s_ = MarkovIBL.return_alternative_item(['B', 'C'], s_)
            alt_r = MarkovIBL.return_alternative_item(REWARD_DICT['B1'], r)
            self.memory.learn(state='<S%d>' % (2), curr_state=alt_s_, next_state=None, response=a_, reward=alt_r)

        self.memory.advance(self.advance_time)

    def encode_memory_punishment(self):
        """
        This will create a biased memory encoding: sensitive only to
        :return:
        """
        s = 'A'
        s_ = self.markov_state._curr_state
        a = self.markov_state.state1_response
        a_ = self.markov_state.state2_response
        r = self.markov_state.received_reward

        if r < 0:
            self.memory.learn(state='<S%d>' % (1), curr_state=s, next_state=s_, response=a, reward=r)
            self.memory.learn(state='<S%d>' % (2), curr_state=s_, next_state=None, response=a_, reward=r)
            # encode alternative state will make hybrid become MB
            alt_s_ = MarkovIBL.return_alternative_item(['B', 'C'], s_)
            alt_r = MarkovIBL.return_alternative_item(REWARD_DICT['B1'], r)
            self.memory.learn(state='<S%d>' % (2), curr_state=alt_s_, next_state=None, response=a_, reward=alt_r)

        self.memory.advance(self.advance_time)

    def encode_memory_alternative(self):
        """
        Learn both current state-reward and alternative state-reward
        This will mae IBL some what like MB
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

        # encode alternative state will make hybrid become MB
        alt_s_ = MarkovIBL.return_alternative_item(['B', 'C'], s_)
        alt_r = MarkovIBL.return_alternative_item(REWARD_DICT['B1'], r)
        self.memory.learn(state='<S%d>' % (2), curr_state=alt_s_, next_state=None, response=a_, reward=alt_r)
        self.memory.advance(self.advance_time)

    def encode_memory_alternative_frequency(self):
        """
        Encode memory that consider common/rare, similar to alternative but in this time
        when common path, encode current reward memory
        when rare path, encode alternative reward memory
        :return:
        """
        s = 'A'
        s_ = self.markov_state._curr_state
        a = self.markov_state.state1_response
        a_ = self.markov_state.state2_response
        r = self.markov_state.received_reward

        self.memory.learn(state='<S%d>' % (1), curr_state=s, next_state=s_, response=a, reward=r)
        # common
        if s_ == self.markov_state._best_blended_state:
            self.memory.learn(state='<S%d>' % (2), curr_state=s_, next_state=None, response=a_, reward=r)
        # rare
        else:
            # encode alternative state will make hybrid become MB
            alt_s_ = MarkovIBL.return_alternative_item(['B', 'C'], s_)
            alt_r = MarkovIBL.return_alternative_item(REWARD_DICT['B1'], r)
            self.memory.learn(state='<S%d>' % (2), curr_state=alt_s_, next_state=None, response=a_, reward=alt_r)
        self.memory.advance(self.advance_time)

    def encode_memory_prediction_error(self):
        """
        Learn the prediction error between blended value and actual reward
        This will make IBL some what like MB
        :return:
        """
        s = 'A'
        s_ = self.markov_state._curr_state
        a = self.markov_state.state1_response
        a_ = self.markov_state.state2_response
        r = self.markov_state.received_reward


        b_value, c_value = self.markov_state._b_value, self.markov_state._b_value
        r1, r0 = REWARD_DICT['B1']

        if (s_ == 'B') and (r > 0):
            learn_b = r1 - b_value
            learn_c = r0 - c_value
        elif (s_ == 'B') and (r <= 0):
            learn_b = r0 - b_value
            learn_c = r1 - c_value
        elif (s_ == 'C') and (r > 0):
            learn_b = r0 - b_value
            learn_c = r1 - c_value
        elif (s_ == 'C') and (r <= 0):
            learn_b = r1 - b_value
            learn_c = r0 - c_value
        else:
            pass
        learn_a = self.lambda_parameter * r
        learn_a, learn_b, learn_c = np.round(learn_a, 2), np.round(learn_b, 2), np.round(learn_c, 2)
        self.memory.learn(state='<S%d>' % (1), curr_state='A', next_state=s_, response=a, reward=learn_a)
        self.memory.learn(state='<S%d>' % (2), curr_state='B', next_state=None, response=a_, reward=learn_b)
        self.memory.learn(state='<S%d>' % (2), curr_state='C', next_state=None, response=a_, reward=learn_c)
        self.memory.advance(self.advance_time)


    # =================================================== #
    # ACT-R MATH FUNCTIONS
    # =================================================== #

    @staticmethod
    def boltzmann(options, values, temperature):
        """Returns a Boltzmann distribution of the probabilities of each option"""
        temperature = max(temperature, 0.01)
        vals = np.array(values) / temperature
        bvals = np.exp(vals) / np.sum(np.exp(vals))
        return dict(zip(options, bvals))

    @staticmethod
    def retrieval_time(activation, fixed_cost=.585, F=.63):
        """

        :param fixed_cost: perception and encoding
        :param F: :lf parameter in ACT-R, default =.63
        :param activation:
        :return: retrieval time
        """
        return fixed_cost + F * np.exp(-activation)

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

    def estimate_LL(self):
        """
        LL += np.log(prob)
        :return:
        """
        assert (self.kind in ('markov-rl-mf', 'markov-rl-mb'))
        return np.sum([np.log(s._state1_p) for s in self.log])

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


import scipy.optimize as opt
class MarkovSimulation():
    GROUP_VAR = ['pre_received_reward', 'pre_state_frequency']

    @staticmethod
    def run_single_simulation(model='markov-ibl-mb', n=200, verbose=False, **params):

        m = MarkovIBL(model=model, verbose=verbose, **params)
        m.memory.activation_history = []
        m.run_experiment(n=n)
        return m

    @staticmethod
    def run_simulations(model='markov-ibl-mb', e=1, n=200, verbose=False, **params):
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

class MarkovEstimateion():
    alpha = .2
    temperature = .2
    response_name = 'state1_response'
    feedback_name = 'received_reward'

    def __init__(self):
        self.data = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @staticmethod
    def load_subject_data(subject_dir, subject_id=None):
        df = pd.concat([pd.read_csv(f, index_col=0) for f in glob.glob(os.path.join(subject_dir, '*', '', 'test.csv'))], axis=0)
        try:
            return df[df['subject_id']=='sub%s' % (subject_id)]
        except:
            return df

    @staticmethod
    def boltzmann(options, values, temperature):
        """Returns a Boltzmann distribution of the probabilities of each option"""
        temperature = max(temperature, 0.01)
        vals = np.array(values) / temperature
        bvals = np.exp(vals) / np.sum(np.exp(vals))
        return dict(zip(options, bvals))

    @staticmethod
    def estimate_LL(data, alpha, temperature):
        """For each trial, calculate the probability of that response, sum the log likelihoods, and update the values"""
        choices = list(set(data[MarkovEstimateion.response_name]))
        Q = dict(zip(choices, [0 for x in choices]))
        LL = 0.0
        for response, feedback in zip(data[MarkovEstimateion.response_name], data[MarkovEstimateion.feedback_name]):
            # Calculate log likelihood of response
            options = Q.keys()
            # TODO: can we replace this value with blended value?
            # does it make sense to pass in blended value?
            values = [Q[opt] for opt in options]
            prob = MarkovEstimateion.boltzmann(options, values, temperature)[response]

            # Sum up the LLs
            LL += np.log(prob)

            # Updates the Q values using Q-learning
            Q_old = Q[response]
            reward = feedback  # RLEstimation.rewards[feedback]
            Q[response] = Q_old + alpha * (reward - Q_old)
        return LL

    def vLLproc(self, array):
        """Vector function of data"""
        alpha = array[0]
        temp = array[1]
        return -1 * MarkovEstimateion.estimate_LL(self.data, alpha, temp)