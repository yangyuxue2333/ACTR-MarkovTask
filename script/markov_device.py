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
import actr
import random
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from functools import reduce
import scipy.optimize as opt


STATE = (0, 1, 2)
TEXT = ("", "A1", "A2", "B1", "B2", "C1", "C2")
COLOR = ("GREEN", "RED", "BLUE")
LOCATION = ("LEFT", "RIGHT")
RESPONSE_MAP = [{'f':'A1', 'k':'A2'},
                {'f':'B1', 'k':'B2'},
                {'f':'C1', 'k':'C2'}]
REWARD = {'B1': 2,
          'B2': 0,
          'C1': 0,
          'C2': 2}
PROBABILITY = {'MARKOV_PROBABILITY':.7, 'REWARD_PROBABILITY':.7}


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
        self.reward_probability = PROBABILITY['REWARD_PROBABILITY']
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

        self.state0()

    @property
    def state1_response_time(self):
        return self._state1_response_time

    @property
    def state2_response_time(self):
        return self._state2_response_time

    @state1_response_time.setter
    def state1_response_time(self, val):
        self._state1_response_time = val

    @state2_response_time.setter
    def state2_response_time(self, val):
        self._state2_response_time = val

    def state0(self):
        self.state1_stimuli = MarkovStimulus('A1'), MarkovStimulus('A2')
        self.state = 0

    def state1(self, response):

        self.state1_response = response
        self.state1_selected_stimulus = RESPONSE_MAP[0][response]

        if RESPONSE_MAP[0][response] == 'A1':
            if random.random() < self.markov_probability:
                self.state2_stimuli = MarkovStimulus('B1'), MarkovStimulus('B2')
                self.state_frequency = 'common'
            else:
                self.state2_stimuli = MarkovStimulus('C1'), MarkovStimulus('C2')
                self.state_frequency = 'rare'
            self.state2_selected_stimulus = RESPONSE_MAP[1][response]

        if RESPONSE_MAP[0][response] == 'A2':
            if random.random() < self.markov_probability:
                self.state2_stimuli = MarkovStimulus('C1'), MarkovStimulus('C2')
                self.state_frequency = 'common'
            else:
                self.state2_stimuli = MarkovStimulus('B1'), MarkovStimulus('B2')
                self.state_frequency = 'rare'
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
        if random.random() < self.reward_probability:
            self.received_reward = REWARD[self.state2_selected_stimulus]
            self.reward_frequency = 'common'
        else:
            self.received_reward = 0
            self.reward_frequency = 'rare'

        # TODO: may change later depending on reward rule
        if REWARD[self.state2_selected_stimulus] == 0:
            self.reward_frequency = 'common'
        self.state = 3

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

        # init markov state
        self.markov_state = None

    def setup(self, model="markov-model1", actr_params=None, reload=True, verbose=False):
        # TODO: add function to modify parameter set
        self.model = model
        self.verbose = verbose
        self.actr_parameters = actr_params
        self.task_parameters = PROBABILITY.copy()

        # init working dir
        curr_dir = os.path.dirname(os.path.realpath('__file__'))

        self.add_actr_commands()
        if reload:
            # schedule event of detect production/reward before loading model
            # note: if not load, no need to schedule it again
            # actr.schedule_event_now("detect-produc            # actr.schedule_event_now("detect-production-hook")tion-hook")
            # actr.schedule_event_now("detect-reward-hook")

            # load model
            actr.load_act_r_model(os.path.join(curr_dir, "markov-core.lisp"))
            actr.load_act_r_model(os.path.join(curr_dir, model + ".lisp"))
            actr.load_act_r_model(os.path.join(curr_dir, "markov-memory.lisp"))

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

    def remove_actr_commands(self):
        actr.remove_command("markov-update-state0")
        actr.remove_command("markov-update-state1")
        actr.remove_command("markov-update-state2")
        actr.remove_command("markov-update-state3")
        actr.remove_command("markov-key-press")
        actr.remove_command("markov-update-end")

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
            self.log.append(self.markov_state)
            if self.verbose: print(self.markov_state)

    def update_state0(self):
        self.markov_state = MarkovState()
        actr.clear_exp_window()
        actr.add_text_to_exp_window(self.window, '+', x=200, y=100, color='black', font_size=50)

    def update_state1(self):
        actr.clear_exp_window()
        state1 = actr.add_visicon_features(
            ['isa', ['markov-stimulus-location', 'markov-stimulus'],
             'kind', 'markov-stimulus',
             'stage', 1,
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

    def run_experiment(self, n=2):
        """
        run ACT-R model
            m = MarkovACTR(setup=False)
            m.setup(reload=True)
            m.run_experiment(2)
        """

        for i in range(n):
            actr.schedule_event_relative(.01 + 10 * i, "markov-update-state0")
            actr.schedule_event_relative(1 + 10 * i, "markov-update-state1")

            self.index += 1

        # "done"
        actr.schedule_event_relative(20 + 10 * i, "markov-update-end")
        actr.run(30 + 10 * i)

        # clear commands
        self.remove_actr_commands()



    # =================================================== #
    # STATS
    # =================================================== #

    def df_behaviors(self):
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

    # =================================================== #
    # ACTR CHUNK INFO
    # =================================================== #
    def extract_chunk_parameter(self, chunk_name, parameter_name):
        """
        This function will extract the parameter value of a chunk during model running
        """
        try:
            actr.hide_output()
            value = actr.sdp(chunk_name, parameter_name)[0][0]
            actr.unhide_output()
            return (chunk_name, parameter_name, value)
        except:
            print('ERROR: WRONG', chunk_name, parameter_name)


    def __str__(self):
        header = "######### SETUP MODEL " + self.model + " #########"
        task_parameter_info = ">> TASK PARAMETERS: " + str(self.task_parameters) + " <<"
        actr_parameter_info = ">> ACT-R PARAMETERS: " + str(self.actr_parameters) + " <<"
        return "%s\n \t%s\n \t%s\n" % (header, task_parameter_info, actr_parameter_info)

    def __repr__(self):
        return self.__str__()


