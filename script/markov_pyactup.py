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

    # @property
    # def best_blend(self):
    #     if self._curr_state == 1:
    #         return self._state1_best_blend
    #     elif self._curr_state == 2:
    #         return self._state2_best_blend
    #     else:
    #         return
    #
    # @best_blend.setter
    # def best_blend(self, V):
    #     if self._curr_state == 'A':
    #         self._state1_best_blend = V
    #     if self._curr_state in ('B', 'C'):
    #         self._state2_best_blend = V

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


class MarkovHuman(MarkovState):
    """A class for recording a Markov trial"""

    def __init__(self, model='markov-monkey', verbose=True):
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
                                'REWARD': REWARD_DICT}

        # init pseudo_random_table
        self.init_pseudo_random_tables()




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
            self.update_random_walk_reward_probabilities()
            self.markov_state.reward()

        # log
        if self.markov_state.state == 3:

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
        return pd.DataFrame(rows, columns=['state1_response',
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
                                 **s.curr_reward_probability_dict} for s in self.log])
        df_wide = df_wide.reset_index()
        res = df_wide.melt(id_vars=['index', 'state2_selected_stimulus', 'received_reward'],
                           var_name='state2_stimulus', value_name='reward_probability').sort_values(by='index')
        return res

    def df_optimal_behaviors(self, num_bin=4):
        """
        Estimate the optimal response
        First, look at selected responses from state2, and curr_reward_probability_dict.
        If random walk is enabled, curr_reward_probability_dict = random
        Otherwise, curr_reward_probability_dict = fixed
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

    def calculate_stay_probability(self):
        """
        Calculate the probability of stay:
            A trial is marked as "STAY" if the agent selects the same action in current trial (e.g. LEFT)
            as the previous trial
        """
        df = self.df_behaviors()
        df['state1_stay'] = df['state1_response'].shift(-1)
        df['state1_stay'] = df.apply(
            lambda x: 1 if x['state1_stay'] == x['state1_response'] else (np.nan if pd.isnull(x['state1_stay']) else 0),
            axis=1)
        return df

    def df_postprocess_behaviors(self, state1_response='f', state2_response='k'):
        df = pd.merge(self.df_behaviors(), self.df_optimal_behaviors())
        # df['index_bin'] = pd.cut(df['index'], 10, labels=False, ordered=False, right=False)
        df['received_reward_norm'] = df['received_reward'] / df['received_reward'].max()
        df['received_reward_sum'] = df['received_reward_norm'].cumsum()
        df['is_optimal'] = df.apply(lambda x: 1 if (
                    x['state1_response'] == x['state1_optimal'] and x['state2_response'] == x['state2_optimal']) else 0,
                                    axis=1)
        # df['is_optimal'] = df.apply(lambda x: 1 if (x['state1_response'] == state1_response and x['state2_response'] == state2_response) else 0, axis=1)
        df['received_reward_sum_prop'] = df.apply(lambda x: x['received_reward_sum'] / ((x['index'] + 1)), axis=1)
        df = pd.merge(df, df.groupby(['index_bin'])['is_optimal'].mean().reset_index(), how='left', on='index_bin',
                      suffixes=('', '_mean'))
        df['optimal_response_sum'] = df['is_optimal'].cumsum()
        df['optimal_response_sum_prop'] = df.apply(lambda x: x['optimal_response_sum'] / ((x['index'] + 1)), axis=1)
        return df

    def __str__(self):
        header = "######### SETUP MODEL " + self.kind + " #########\n" + str(self.task_parameters)
        return header

    def __repr__(self):
        return self.__str__()


class MarkovIBL(MarkovState):
    """A class for recording a Markov trial"""

    def __init__(self, model='markov-rlmf', verbose=True, **params):
        """Inits a markov trial
        stimuli contain a list of markov states, e.g. [("A1", "A2"), ("B1", "B2")]
        """
        assert (model in ('markov-ibl', 'markov-rlmf', 'markov-rlmb'))
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

        # init parameters
        self.rl_parameters = {}
        self.task_parameters = {'MARKOV_PROBABILITY': 0.7,
                                'REWARD_PROBABILITY': {'B1': 0, 'B2': 0, 'C1': 0, 'C2': 0},
                                'REWARD': REWARD_DICT}

        # init pseudo_random_table
        self.init_pseudo_random_tables()

        # init state
        self.markov_state = MarkovState()
        # init IBL memory
        self.memory = pau.Memory(**params)
        self.init_memory()

        # RL parameters
        # RL MF/MB parameters
        self.alpha = .5          # learning rate
        self.beta = 1            # exploration parameter
        self.beta_1mf = 5
        self.beta_1mb = 5
        self.beta_2 = 5
        self.p_parameter = 0    # perseveration parameter
        self.temperature = self.memory.noise # noise parameter
        self.lambda_parameter = .6

        self.q = {(s, a): 0 for s in self.state_space for a in self.action_space}
        self.p = {(s, a): 0 for s in self.state_space for a in self.action_space}
        self.LL = 0.0

        if self.verbose:
            print(self.__str__())


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

    def respond_to_key_press(self):
        key = None
        if self.kind == 'markov-ibl':
            key = self.choose_ibl()
        elif self.kind =='markov-rlmf':
            key = self.choose_rlmf()
        elif self.kind =='markov-rlmb':
            key = self.choose_rlmb()
        else:
            print('error model', self.kind )

        self.response = key
        self.next_state(key)

    def evaluate_ibl(self):
        # b_value, c_value = [np.round(self.memory.blend("reward", curr_state=state), 4) for state in ['B', 'C']]
        # best_val = np.max([b_value, c_value])

        # decide best option
        # state_blend_dict = dict(zip(['B', 'C'], [b_value, c_value]))
        # best_blended_state = random.choice([k for k, v in state_blend_dict.items() if v == best_val])

        best_blended_state, best_blended_val = self.memory.best_blend("reward", ({"curr_state": state} for state in ("B", "C")))
        best_blended_state = best_blended_state['curr_state']
        retrieved_memory = self.memory.retrieve(rehearse=False, curr_state='A', next_state=best_blended_state)
        if retrieved_memory['reward'] > 0:
            a = retrieved_memory['response']
        else:
            a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])
        self.memory.advance()
        self.markov_state._best_blended_state = best_blended_state
        self.markov_state._state_blend_dict = {best_blended_state:best_blended_val}
        return a

    def choose_ibl(self):
        """
        Use IBL algorithm
        :return:
        """
        if self.markov_state._curr_stage == '1':
            a = self.evaluate_ibl()
            return a
        else:
            # a = self.memory.retrieve(curr_state=self.markov_state._curr_state, reward=self._r1)['response']
            retrieved_memory = self.memory.retrieve(rehearse=False, curr_state=self.markov_state._curr_state)
            if retrieved_memory['reward'] > 0:
                a = retrieved_memory['response']
            else:
                a = random.choice([action for action in self.action_space if action != retrieved_memory['response']])
            self.memory.advance()
            return a

    def get_state2_choice(self):
        assert self.markov_state._curr_stage == '2'
        """Get simulated choice at state2.
        Keyword arguments:
        q: dict of final-state action values
        beta: exploration parameter
        """
        q = self.q.copy()
        probs = np.array(
            [np.exp(self.beta_2 * v) for (s, a), v in q.items() if s == self.markov_state._curr_state])
        probs /= np.sum(probs)
        r = random.random()
        s = 0
        for a, x in enumerate(probs):
            s += x
            if s >= r:
                return self.action_space[a]
        return a

    def evaluate_rlmf(self):
        assert self.markov_state._curr_stage == '1'
        q = self.q.copy()
        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)

        rep = 1 if prev_choice == self.action_space[0] else -1
        p = expit(self.beta_1mf * (
                    q[('A', self.action_space[0])] - q[('A', self.action_space[1])] + rep * self.p_parameter))
        self.markov_state._state1_p = p
        return p

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
            return self.get_state2_choice()

    def evaluate_rlmb(self):
        assert self.markov_state._curr_stage == '1'
        q = self.q.copy()
        b_value = max([q[('B', a)] for a in self.action_space])
        c_value = max([q[('C', a)] for a in self.action_space])
        # Determine the choice
        if COMMON_TRANS[self.action_space[1]] == 'B':
            cv = (2 * .7 - 1) * (b_value - c_value)
        else:
            cv = (2 * .7 - 1) * (c_value - b_value)

        try:
            prev_choice = self.log[-1].state1_response
        except:
            prev_choice = random.choice(self.action_space)

        rep = 1 if prev_choice == self.action_space[1] else -1
        p = expit(self.beta_1mb * (cv + rep * self.p_parameter)) # p_right
        self.markov_state._state1_p = p
        return p

    def choose_rlmb(self):
        """
        Use model-base algorithm modified based on Feher da Silva 2018
        :return:
        """
        q = self.q.copy()
        if self.markov_state._curr_stage == '1':
            p = self.evaluate_rlmb()
            if random.random() < p:
                a = self.action_space[1]
            else:
                a = self.action_space[0]
            return a
        else:
            return self.get_state2_choice()

    def encode_memory(self):
        s = 'A'
        s_ = self.markov_state._curr_state
        a = self.markov_state.state1_response
        a_ = self.markov_state.state2_response
        r = self.markov_state.received_reward
        # b_value = self.markov_state._b
        # if r > 0:
        self.memory.retrieve(rehearse=True, state='<S%d>' % (1), curr_state=s, next_state=s_, response=a, reward=r)
        self.memory.retrieve(rehearse=True, state='<S%d>' % (2), curr_state=s_, next_state=None, response=a_, reward=r)
        self.memory.advance(15)

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
            if self.kind == 'markov-ibl':
                self.encode_memory()
            elif self.kind == 'markov-rlmf':
                self.update_q()
            elif self.kind == 'markov-rlmb':
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

    @staticmethod
    def boltzmann(options, values, temperature):
        """Returns a Boltzmann distribution of the probabilities of each option"""
        temperature = max(temperature, 0.01)
        vals = np.array(values) / temperature
        bvals = np.exp(vals) / np.sum(np.exp(vals))
        return dict(zip(options, bvals))

    def update_q(self):
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

    def run_experiment(self, n=1):
        """
        """
        for i in range(n):
            self.markov_state = MarkovState()
            self.markov_state.state0()  # init
            self.respond_to_key_press()
            self.respond_to_key_press()
            self.index += 1

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
        df = pd.DataFrame({**s._state_blend_dict, 'best_blended_state':s._best_blended_state} for s in self.log)
        df.columns = ['state1_blended_b', 'state1_blended_c', 'best_blended_state']
        return df

    def df_q_table(self):
        df_q =pd.DataFrame([s.q for s in self.log])
        df_q.columns = [(s, RESPONSE_CODE[a]) for (s, a) in df_q.columns]
        return df_q
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

    def __str__(self):
        header = "######### SETUP MODEL " + self.kind + " #########\n" + str(self.task_parameters)
        return header

    def __repr__(self):
        return self.__str__()



class MarkovSimulation():
    GROUP_VAR = ['pre_received_reward', 'pre_state_frequency']

    @staticmethod
    def run_single_simulation(model='markov-ibl', n=200, verbose=False, **params):

        m = MarkovIBL(model=model, verbose=verbose, **params)
        m.memory.activation_history = []
        m.run_experiment(n=n)
        return m

    @staticmethod
    def run_simulations(model='markov-ibl', e=1, n=200, verbose=False, **params):
        df_list = []
        for i in tqdm(range(e)):
            m = MarkovIBL(model=model, verbose=verbose, **params)
            m.run_experiment(n=n)
            temp = m.calculate_stay_probability()
            temp['epoch'] = i
            df_list.append(temp)
        res = pd.concat(df_list, axis=0)
        for k,v in params.items():
            res[k]  = v

        res = res.groupby(['epoch'] +
                          MarkovSimulation.GROUP_VAR).\
            agg(state1_stay_mean=('state1_stay', 'mean'),
              state1_response_time_mean=('state1_response_time', 'mean'),
              state2_response_time_mean=('state2_response_time', 'mean')).reset_index()
        return res