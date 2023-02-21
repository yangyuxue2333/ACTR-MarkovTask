import os
import shutil
from markov_device import *
from datetime import date
import time
import glob
from scipy.stats import norm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

global convergence
convergence = 100


# =================================================== #
# SIMULATION CLASS
# =================================================== #

class Simulation:
    @staticmethod
    def simulate(model="markov-model1", n=20, task_params=None, actr_params=None, thresh=None, verbose=False):
        """
        Simulate markov model
        @param: thresh determines whether the model learns optimal action sequence
        Return: a ACT-R model object

            example:
            m = simulate(model="markov-model1",
                        n=200,
                        task_params={'M':1},
                        actr_params={'v': 't'},
                        thresh=0,
                        verbose=True)
        """
        global convergence
        if convergence < 0:
            print('>>> Failed to converge <<<')
            return
        m = MarkovACTR(setup=False)
        m.setup(model=model, actr_params=actr_params, task_params=task_params, reload=True, verbose=verbose)
        m.run_experiment(n)
        if not thresh:
            return m

        # threshold based on overall stay
        # if stay >= 1, exclude this simulation
        # otherwise, keep this
        df_stay = m.calculate_stay_probability()
        perf = df_stay['state1_stay'].mean()


        # threshold based on optimal performance
        # df = m.df_postprocess_behaviors()
        # perf = df['optimal_response_sum_prop'].loc[len(df) - 1]

        if perf < thresh:
            if verbose: print(m)
            return m
        else:
            print('>>> bad simulation %.2f <<< [threshold = %.2f]' % (perf, thresh))
            convergence -= 1
            return Simulation.simulate(model, n, task_params, actr_params, thresh)

    @staticmethod
    def simulate_stay_probability(model="markov-model1", epoch=1, n=20, task_params=None, actr_params=None, log=False, thresh=None, verbose=False):
        rewards = 0.0
        beh_list, state1stay_list, utrace_list, atrace_list = [], [], [], []

        start_time = time.time()
        # for i in tqdm(range(epoch)):
        for i in range(epoch):
            m = Simulation.simulate(model=model, n=n, task_params=task_params, actr_params=actr_params, thresh=thresh, verbose=verbose)
            if (i==0 and verbose): print(m)

            # stay probability
            state1stay = m.calculate_stay_probability()
            utrace, atrace = m.df_postprocess_actr_traces()
            beh = m.df_postprocess_behaviors()
            state1stay['epoch'] = i
            utrace['epoch'] = i
            atrace['epoch'] = i
            beh['epoch'] = i

            # subset
            # state1stay = state1stay.groupby(['epoch', 'pre_received_reward', 'state_frequency']).agg(
            #     {'state1_stay': lambda x: x.mean(skipna=True)}).reset_index()


            if log:
                df_beh, df_state1stay, df_utrace, df_atrace = beh, state1stay, utrace, atrace
                file_path = Simulation.save_simulation(beh, dir_path=log, file_name=model+'-sim-logdata')
                Simulation.save_simulation(state1stay, dir_path=log, file_name=model+'-sim-staydata', verbose=verbose)
                Simulation.save_simulation(utrace, dir_path=log, file_name=model+'-actr-udata', verbose=verbose)
                Simulation.save_simulation(atrace, dir_path=log, file_name=model+'-actr-adata',verbose=verbose)

            else:
                beh_list.append(beh)
                state1stay_list.append(state1stay)
                utrace_list.append(utrace)
                atrace_list.append(atrace)

                # plot
                rewards += beh['received_reward'].sum()/len(beh)

            # show estimated time
            if (i==0):
                duration = (epoch * (time.time() - start_time)) / 60
                print("...ESTIMATED RUN TIME [%.2f] (min)..." % (duration))


        if log:
            Simulation.save_params(dir_path=log,
                                   epoch=epoch,
                                   n=n,
                                   actr_params=actr_params,
                                   task_params=task_params,
                                   file_path=file_path,
                                   verbose=verbose)
            return df_beh, df_state1stay, df_utrace, df_atrace
        else:
            # merge df list
            df_beh = pd.concat(beh_list, axis=0)
            df_state1stay = pd.concat(state1stay_list, axis=0)#.groupby(['received_reward', 'reward_frequency', 'state_frequency']).agg({'state1_stay': lambda x: x.mean(skipna=True)}).reset_index()
            df_utrace = pd.concat(utrace_list, axis=0)#.groupby(['index_bin', 'action', 'state', 'response']).agg({':utility': lambda x: x.max(skipna=True)}).reset_index()
            df_atrace = pd.concat(atrace_list, axis=0)#.groupby(['index_bin', 'memory', 'memory_type']).agg({':Reference-Count': lambda x: x.max(skipna=True), ':Last-Retrieval-Activation': lambda x: x.max(skipna=True)}).reset_index()

            # calculate expected reward
            # mean reward probabilities for 4 state2 responses
            # expected reward = max_reward_probability * reward gained (2 default)
            max_reward_probability = df_beh[['reward_probability_B1',
                                             'reward_probability_B2',
                                             'reward_probability_C1',
                                             'reward_probability_C2']].mean().max()
            expected_reward = max_reward_probability * np.max(list(m.task_parameters['REWARD'].values()))
            # max_id = np.argmax(list(m.task_parameters['REWARD_PROBABILITY'].values()))
            # expected_reward = list(m.task_parameters['REWARD'].values())[max_id] * \
            #                   list(m.task_parameters['REWARD_PROBABILITY'].values())[max_id]
            print('>>>>>>>>> SIMULATION REWARD GAINED <<<<<<<<<< \t EPOCH:', epoch)
            print('GAINED R: %.2f (EXPECTED R: %.2f) \t [%.2f %%]\n\n\n\n' % (
            (rewards / epoch), expected_reward, 100 * (rewards / epoch) / expected_reward))
            print("...RUNNING TIME [%.2f] (s)..." % (time.time() - start_time))

            return df_beh, df_state1stay, df_utrace, df_atrace


    # =================================================== #
    # RECORD SIMULATED DATA
    # =================================================== #

    @staticmethod
    def save_simulation(df, dir_path, file_name, verbose=False):
        """
        parent_dir = '..xx/ACTR-MarkovTask'
        dir_path = 'data/model/param_simulation_xx'
        """
        parent_dir = os.path.dirname(os.getcwd())
        data_dir_path = os.path.join(parent_dir, dir_path)
        if not os.path.exists(data_dir_path):
            os.makedirs(data_dir_path)
        today = date.today().strftime('-%m%d%y') #.strftime('-%m-%d-%Y')
        # file_path=data_dir_path+file_name+today+'.csv'
        file_path =  os.path.join(data_dir_path, file_name+'.csv') #data_dir_path + file_name + '.csv'
        mode='w'
        header = True
        if os.path.exists(file_path):
            mode='a'
            header = False
        df.to_csv(file_path, mode=mode, header=header)
        if verbose: print('...START LOG SIMULATION OUTPUTS.../n/t', file_name)
        return file_path

    @staticmethod
    def save_params(dir_path, epoch, n, actr_params, task_params, file_path, verbose=False):
        """
        parent_dir = '..xx/ACTR-MarkovTask'
        dir_path = 'data/model/param_simulation_xx'
        """
        parent_dir = os.path.dirname(os.getcwd())
        log_path = os.path.join(parent_dir, dir_path, 'log.csv')

        param_dict={'epoch':epoch, 'n':n, **actr_params, **task_params,
                    'model_name':file_path.split('/')[-1].split('-')[1][-1],
                    'param_id':file_path.split('/')[-2],
                    'file_path':file_path,}
        df = pd.DataFrame(param_dict.values(), index=param_dict.keys()).T
        mode='w'
        header=True

        if os.path.exists(log_path):
            mode='a'
            header=False

        df.to_csv(log_path, mode=mode, header=header, index=True)
        if verbose: print('...START LOG SIMULATION PARAMETERS.../n/t..', log_path)

    @staticmethod
    def try_simulation_example():
        """
        try simulation exaple
        """
        # simulation parameter:
        e = 1  # e: the number of simulation epoch
        n = 20  # n: the number of trials per simulas

        # log parameter:
        main_dir = os.path.dirname(os.getcwd())
        log_dir = 'data/param_simulation_test/'  # destination dir
        model_name = 'markov-model1'  # model name

        # task parameter combination
        m = [0, 0.5, 1, 1.5]
        r = [0.1, 5, 10]
        random_walk = ['LOAD']  # use pre-loaded reward probability in /fixed

        # actr parameter combination
        ans = [0.2, 0.7]
        egs = [0.2, 0.7]
        alpha = [0.2, 0.7]
        lf = [0.5, 1]

        task_param_set = list(itertools.product(*[random_walk, m, r]))[:2]
        actr_param_set = list(itertools.product(*[ans, egs, alpha, lf]))[:1]

        print('TOTAL NUM PARAMETER COMBINATION [%d] \n\t[TASK PARAM: (%d)], \n\t[ACT-R PARAM: (%d)]' % (
            len(task_param_set) * len(actr_param_set), len(task_param_set), len(actr_param_set)))

        start_time = time.time()

        for i in range(len(task_param_set)):

            random_walk, m, r = task_param_set[i]
            task_params = {'REWARD': {'B1': r, 'B2': r, 'C1': r, 'C2': r}, 'RANDOM_WALK': random_walk, 'M': m}

            for j in range(len(actr_param_set)):

                ans, egs, alpha, lf = actr_param_set[j]
                actr_params = {'ans': ans, 'egs': egs, 'alpha': alpha, 'lf': lf, 'bll': 0.5, 'mas': 2}

                param_folder_id = 'param_task%d_actr%d/' % (i, j)

                # check if alreay simulated
                # log_dir = '%s%s' % (log_dir, param_folder_id)
                dest_dir = os.path.join(main_dir, log_dir, param_folder_id)
                if not Simulation.simple_check_exist(dest_dir=dest_dir,
                                                     target_num_files=5,
                                                     verbose=True,
                                                     overwrite=False,
                                                     special_suffix="*"):
                    Simulation.simulate_stay_probability(model=model_name,
                                                         epoch=e,
                                                         n=n,
                                                         task_params=task_params,
                                                         actr_params=actr_params,
                                                         log=dest_dir,
                                                         verbose=False)
                    print("\t...COMPLETE...%s" % (param_folder_id))
                else:
                    print("\t...SKIP ....%s" % (param_folder_id))

                if (i == 0) and (j == 0):
                    end_time = time.time()
                    print('>>> ESTIMATED ONE COMBINATION SET: RUNNING TIME (EST): [%.2f] minutes <<<' % (
                            (1 / 60) * len(task_param_set) * len(actr_param_set) * (end_time - start_time)))
        print('>>> FINISHED: RUNNING TIME: [%.2f] <<<' % (time.time() - start_time))

    # =================================================== #
    # LOAD SIMULATED DATA
    # =================================================== #

    @staticmethod
    def load_simulation(data_path='data/param_simulation_1114/param_task0_actr0', model_name='markov-model1', index_thres=None, verbose=True, load_agg=False):
        """
        Load simulated data for a single parameter set
        :param data_path: 'data/model/param_simulation_xx/param_task0_actr0'
        :param model_name: require full name of model e
        :param index_thres:
        :param verbose:
        :return:
        parent_dir = '..xx/ACTR-MarkovTask'
        """
        parent_dir = os.path.dirname(os.getcwd())
        data_path = os.path.join(parent_dir, data_path)

        df1_utrace = pd.read_csv(os.path.join(data_path, model_name + '-actr-udata.csv'))
        df1_utrace[':utility'] = df1_utrace[':utility'].apply(pd.to_numeric, errors='coerce')

        df1_atrace = pd.read_csv(os.path.join(data_path, model_name + '-actr-adata.csv'))
        df1_atrace[':Reference-Count'] = df1_atrace[':Reference-Count'].apply(pd.to_numeric, errors='coerce')
        df1_atrace[':Activation'] = df1_atrace[':Activation'].apply(pd.to_numeric, errors='coerce')
        df1_atrace[':Last-Retrieval-Activation'] = df1_atrace[':Last-Retrieval-Activation'].apply(pd.to_numeric, errors='coerce')
        param_log = pd.read_csv(os.path.join(data_path, 'log.csv')).loc[0]

        if verbose:
            print('...SUCCESSFULLY LOADED DATA...AGGREGATE[%s]' % (load_agg))
            print(param_log)

        if load_agg:
            df_agg = pd.read_csv(os.path.join(data_path, 'aggregate', model_name + '-agg.csv')).apply(pd.to_numeric, errors='ignore')
            return df_agg, df1_utrace, df1_atrace
        else:
            df1 = pd.read_csv(os.path.join(data_path, model_name + '-sim-logdata.csv')).dropna(axis=0).apply(pd.to_numeric, errors='ignore')
            df1_state1stay = pd.read_csv(os.path.join(data_path, model_name + '-sim-staydata.csv')).dropna(axis=0).apply(pd.to_numeric, errors='ignore')

            # temporarily fix 0115 simulation stay probability errors
            if 'pre_received_reward' not in df1_state1stay.columns:
                print('temporarily fix 0115 simulation stay probability errors')
                df1_state1stay = Simulation.temporary_update_stay_probability(df1)

            if index_thres:
                df1 = df1[df1['index'] < index_thres]
                df1_state1stay = df1_state1stay[df1_state1stay['index'] < index_thres]
            return df1, df1_state1stay, df1_utrace, df1_atrace

    # =================================================== #
    # HELPER FUNCTIONS
    # =================================================== #
    @staticmethod
    def check_parameters(log_file_path, task_param_set, actr_param_set, epoch, n, model_name, param_id):
        if not os.path.exists(log_file_path):
            return False
        log = pd.read_csv(log_file_path, header=0, index_col=0).drop(columns=['file_path'])
        log_param_list = log.to_records(index=False).tolist()
        for log_param_set in log_param_list:
            curr_param_set = (epoch, n,
                              actr_param_set['seed'],
                              actr_param_set['ans'],
                              actr_param_set['egs'],
                              actr_param_set['alpha'],
                              # actr_param_set['v'],
                              actr_param_set['lf'],
                              actr_param_set['bll'],
                              actr_param_set['mas'],
                              str(task_param_set['REWARD']),
                              task_param_set['RANDOM_WALK'],
                              task_param_set['M'],
                              int(model_name[-1]),
                              param_id.split("/")[0])

            if log_param_set == curr_param_set:
                # print('Found...')
                return True
            # print('No...', log_param_set, curr_param_set)
        return False

    @staticmethod
    def simple_check_exist(dest_dir, target_num_files, verbose=True, overwrite=False, special_suffix="*"):
        """
        Check whether simulation exist using the simplest way
        :param log_dir:
        :return:
        """
        check_msg = ''
        if overwrite:
            shutil.rmtree(dest_dir, ignore_errors=True)
            check_msg += '...OVERWRITE...'
        dest_files = os.path.join(dest_dir, '%s.csv' % (special_suffix))
        num_dest_files = glob.glob(dest_files)
        if len(num_dest_files) == target_num_files:
            check_msg += '...SKIP...DIR: [%s]' % (dest_dir)
            pass_check = True
        else:
            check_msg += '...CREATE...[%s]'% (dest_dir)
            os.makedirs(dest_dir, exist_ok=True)
            pass_check = False

        if verbose: print(check_msg)
        return pass_check

    @staticmethod
    def map_func1(x):
        """
        process memory information
        """
        a, b, c, d, e = x['state1_selected_stimulus'], x['state2_selected_stimulus'], x['state1_response'], x[
            'state2_response'], x['received_reward']
        if a[0] == 'A' and b[0] == 'B' and c == 'f':
            return 'A1-%s-%s-%s' % (RESPONSE_CODE[c], b[0] , 'none')
        if a[0] == 'A' and b[0] == 'B' and c == 'k':
            return 'A3-%s-%s-%s' % (RESPONSE_CODE[c], b[0] , 'none')
        if a[0] == 'A' and b[0] == 'C' and c == 'f':
            return 'A2-%s-%s-%s' % (RESPONSE_CODE[c], b[0] , 'none')
        if a[0] == 'A' and b[0] == 'C' and c == 'k':
            return 'A4-%s-%s-%s' % (RESPONSE_CODE[c], b[0] , 'none')

    @staticmethod
    def map_func2(x):
        a, b, c, d, e = x['state1_selected_stimulus'], x['state2_selected_stimulus'], x['state1_response'], x[
            'state2_response'], x['received_reward']
        if e > 0:
            return '%s-%s-%s-%s' % (b, RESPONSE_CODE[d], 'none', 1)
        else:
            return '%s-%s-%s-%s' % (b, RESPONSE_CODE[d], 'none', 0)

    @staticmethod
    def process_memory_data(dfm):
        dfm['state1_memory'] = dfm.apply(Simulation.map_func1, axis=1)
        dfm['state2_memory'] = dfm.apply(Simulation.map_func2, axis=1)

        res = pd.concat([dfm.groupby(['epoch', 'state1_memory'])['index'].count(). \
                        reset_index().rename(columns={'state1_memory': 'memory', 'index': 'trial_count'}),
                         dfm.groupby(['epoch', 'state2_memory'])['index'].count() \
                        .reset_index().rename(columns={'state2_memory': 'memory', 'index': 'trial_count'})], axis=0)
        return res

    @staticmethod
    def caluclate_optimal(df, main_dir, optimal_response='both'):
        """
        Calculate optimal response for subject data
        :param df: test.csv
        :param main_dir:
        :param optimal_response:
        :return:
        """
        assert optimal_response in ('state1', 'state2', 'both')
        df = df.copy()
        df_rp = pd.read_csv(os.path.join(main_dir, 'data/fixed/masterprob4.csv'))

        df_rp.columns = ['B1', 'B2', 'C3', 'C4']
        df_rp['state2_optimal'] = df_rp.apply(lambda x: df_rp.columns[x.argmax()], axis=1)
        df_rp['state1_optimal'] = df_rp['state2_optimal'].map({'B1': 'A1', 'B2': 'A1', 'C3': 'A2', 'C4': 'A2'})
        df_rp['state1_response_optimal'] = df_rp['state1_optimal'].map({'A1': 49.0, 'A2': 48.0})
        df_rp['state2_response_optimal'] = df_rp['state2_optimal'].map({'B1': 49.0, 'B2': 48.0, 'C3': 49.0, 'C4': 48.0})
        dfo = pd.concat([df, df_rp.head(200).drop(index=0)], axis=1)

        dfo['state1_is_optimal'] = dfo.apply(lambda x: 1 if x['state1_response'] == x['state1_response_optimal'] else 0,
                                             axis=1)
        dfo['state2_is_optimal'] = dfo.apply(lambda x: 1 if x['state2_response'] == x['state1_response_optimal'] else 0,
                                             axis=1)

        if optimal_response == 'both':
            dfo['is_optimal'] = dfo.apply(
                lambda x: 1 if (x['state1_is_optimal'] == 1) & (x['state2_is_optimal'] == 1) else 0, axis=1)
        else:
            dfo['is_optimal'] = dfo[optimal_response + '_is_optimal']
        dfo['performance'] = optimal_response

        dfo['received_reward_norm'] = dfo['received_reward'] / dfo['received_reward'].max()
        dfo['received_reward_sum'] = dfo.groupby(['subject_id'])['received_reward_norm'].cumsum()
        dfo['optimal_response_sum'] = dfo.groupby(['subject_id'])['is_optimal'].cumsum()

        dfo['received_reward_sum_prop'] = dfo.apply(lambda x: x['received_reward_sum'] / ((x['index'] + 1)), axis=1)
        dfo['optimal_response_sum_prop'] = dfo.apply(lambda x: x['optimal_response_sum'] / ((x['index'] + 1)), axis=1)
        return dfo

class MaxLogLikelihood:
    MAXLL_FACTOR_VAR = ['pre_received_reward', 'pre_state_frequency'] #['pre_received_reward', 'pre_state_frequency']
    MAXLL_DEP_VAR = ['state1_stay', 'state1_response_time', 'state2_response_time']
    MAXLL_PARAMETERS = ['ans', 'egs', 'alpha', 'lf', 'bll', 'REWARD', 'M']


    # =================================================== #
    # PROCESS SUBJECT DATA
    # =================================================== #
    @staticmethod
    def process_subject_data_teddy(data_dir='data/human/task_data', log=None):
        """
                Load and reformat emperical data (Teddy)
                parent_dir = '..xx/ACTR-MarkovTask'
                data_dir = 'data/human/task_data'
                """
        assert (os.getcwd().split('/')[-1] == 'ACTR-MarkovTask')

        parent_dir = os.path.dirname(os.getcwd())
        data_dir = os.path.join(parent_dir, data_dir)
        df_list = []
        for f in np.sort(glob.glob(data_dir + '/%s' % ('*.txt'))):
            df = pd.read_csv(f, header=None)
            df.columns = ['trial', 'firstState', 'secondState', 'thirdState', 'firstAction', 'secondAction',
                          'firstReactionTime', 'secondReactionTime', 'thirdReactionTime', 'slowFirstAction',
                          'slowSecondAction', 'slowThirdAction',
                          'firstTransition', 'secondTransition', 'HighlyRewardingState', 'rewardPosition', 'rewarded',
                          'totalReward']

            subject_id = f.split('/')[-1].split('.')[0].split('_')[0]
            data_file = f.split('/')[-1].split('.')[0].split('_')[1]
            df.insert(0, "subject_id", subject_id)
            df.insert(0, "data_file", data_file)
            # df['data_file'] = f.split('/')[-1].split('.')[0].split('_')[1]

            # convert column name to Cher's simulation data structure
            df = df.drop(columns=['firstState'])
            df['secondState'] = df['secondState'].map({1: 'A1', 2: 'A2'})
            df['thirdState'] = df['thirdState'].map({1: 'B1', 2: 'B2', 3: 'C1', 4: 'C2'})
            df['firstAction'] = df['firstAction'].map({0: 'f', 1: 'k'})
            df['secondAction'] = df['secondAction'].map({0: 'f', 1: 'k'})
            df['firstTransition'] = df['firstTransition'].map({0: 'common', 1: 'rare'})
            df['secondTransition'] = df['secondTransition'].map({0: 'common', 1: 'rare'})
            df['HighlyRewardingState'] = df['HighlyRewardingState'].map({1: 'B1', 2: 'B2', 3: 'C1', 4: 'C2'})
            df['rewardPosition'] = df['rewardPosition'].map({1: 'B1', 2: 'B2', 3: 'C1', 4: 'C2'})

            # rename columns
            df = df.rename(columns={'trial': 'index',
                                    'secondState': 'state1_selected_stimulus',
                                    'thirdState': 'state2_selected_stimulus',
                                    'firstAction': 'state1_response',
                                    'secondAction': 'state2_response',
                                    'firstReactionTime': 'state1_response_time',
                                    'secondReactionTime': 'state2_response_time',
                                    'thirdReactionTime': 'state2_response_time',
                                    'slowFirstAction': 'slow_first_action',
                                    'slowSecondAction': 'slow_second_action',
                                    'slowThirdAction': 'slow_third_action',
                                    'firstTransition': 'state_frequency',
                                    'secondTransition': 'reward_frequency',
                                    'HighlyRewardingState': 'highly_rewarding_state',
                                    'rewardPosition': 'reward_position',
                                    'rewarded': 'received_reward',
                                    'totalReward': 'total_reward'})

            df_list.append(df)
            if log:
                log_dest_dir = log + '/%s' % (subject_id)
                log_dest_path = '%s/%s.csv' % (log_dest_dir, data_file)
                # print(log_dest_path)
                if not os.path.exists(log_dest_dir):
                    os.makedirs(log_dest_dir)
                df.to_csv(log_dest_path, header=True, index=False)
        dff = pd.concat(df_list, axis=0)
        return dff

    @staticmethod
    def process_subject_data_online(main_dir, raw_subject_dir, new_subject_dir, overwrite=False, verbose=True):
        """
        raw_subject_dir = 'data/human/online_csvs'
        new_subject_dir = 'data/human/subject_data'
        subject_id = 'sub1'
        """
        subject_ids = np.sort([f.split('_')[-1].split('.')[0] for f in os.listdir(os.path.join(main_dir, raw_subject_dir)) if f.endswith('.csv')])
        for subject_id in subject_ids:
            # read raw subject data
            df = pd.read_csv(os.path.join(main_dir, raw_subject_dir, 'mbmf_%s.csv' % (subject_id))).dropna(
                subset='trial_stage', axis=0)
            # select useful columns
            usecols = ['subject_id', 'trial_index', 'practice_trial', 'reward', 'transition', 'transition_type',
                       'key_press', 'trial_stage', 'rt']

            # process practice data
            df_practice = df[df['practice_trial'] == 'practice'][usecols].sort_values(by='trial_index')

            df_practice1 = df_practice[df_practice['trial_stage'] == '1'][['subject_id', 'trial_index', 'key_press', 'rt']]
            df_practice1['index'] = np.arange(len(df_practice1))
            df_practice1 = df_practice1.rename(columns={'key_press': 'state1_response', 'rt': 'state1_response_time'}).drop(columns=['trial_index'])

            df_practice2 = df_practice[df_practice['trial_stage'] == '2'][['subject_id', 'trial_index', 'transition', 'reward', 'key_press', 'rt']]
            df_practice2['index'] = np.arange(len(df_practice2))
            df_practice2 = df_practice2.rename(columns={'key_press': 'state2_response', 'rt': 'state2_response_time', 'reward': 'received_reward',
                         'transition': 'state_frequency'}).drop(columns=['trial_index'])

            # unstack df to wide format
            df_practice_wide = pd.merge(df_practice1, df_practice2, on=['subject_id', 'index'])
            df_practice_wide['pre_received_reward'] = df_practice_wide['received_reward'].shift()
            df_practice_wide['pre_state_frequency'] = df_practice_wide['state_frequency'].shift()
            df_practice_wide['pre_state1_response'] = df_practice_wide['state1_response'].shift()
            df_practice_wide['state1_stay'] = df_practice_wide.apply(
                lambda x: 1 if x['pre_state1_response'] == x['state1_response'] else 0, axis=1)
            df_practice_wide = df_practice_wide.dropna(axis=0)
            df_practice_wide['pre_received_reward'] = df_practice_wide.apply(
                lambda x: 'reward' if int(x['pre_received_reward']) == 1 else 'non-reward', axis=1)

            # process test data
            df_test = df[df['practice_trial'] == 'real'][usecols].sort_values(by='trial_index')
            df_test1 = df_test[df_test['trial_stage'] == '1'][['subject_id', 'trial_index', 'key_press', 'rt']]
            df_test1['index'] = np.arange(len(df_test1))
            df_test1 = df_test1.rename(columns={'key_press': 'state1_response', 'rt': 'state1_response_time'}).drop(
                columns=['trial_index'])

            df_test2 = df_test[df_test['trial_stage'] == '2'][
                ['subject_id', 'trial_index', 'transition', 'reward', 'key_press', 'rt']]
            df_test2['index'] = np.arange(len(df_test2))
            df_test2 = df_test2.rename(
                columns={'key_press': 'state2_response', 'rt': 'state2_response_time', 'reward': 'received_reward',
                         'transition': 'state_frequency'}).drop(columns=['trial_index'])

            # unstack data to wide format
            df_test_wide = pd.merge(df_test1, df_test2, on=['subject_id', 'index'])
            df_test_wide['pre_received_reward'] = df_test_wide['received_reward'].shift()
            df_test_wide['pre_state_frequency'] = df_test_wide['state_frequency'].shift()
            df_test_wide['pre_state1_response'] = df_test_wide['state1_response'].shift()
            df_test_wide['state1_stay'] = df_test_wide.apply(
                lambda x: 1 if x['pre_state1_response'] == x['state1_response'] else 0, axis=1)
            df_test_wide = df_test_wide.dropna(axis=0)
            df_test_wide['pre_received_reward'] = df_test_wide.apply(
                lambda x: 'reward' if int(x['pre_received_reward']) == 1 else 'non-reward', axis=1)

            #reorder columns
            col_order = ['subject_id', 'index', 'state_frequency', 'received_reward',
                         'pre_received_reward', 'pre_state_frequency',
                         'state1_response', 'state2_response', 'state1_stay', 'state1_response_time', 'state2_response_time']
            df_practice_wide = df_practice_wide[col_order]
            df_test_wide = df_test_wide[col_order]

            dest_dir = os.path.join(main_dir, new_subject_dir, subject_id)
            if not Simulation.simple_check_exist(dest_dir, target_num_files=2, verbose=True, overwrite=overwrite):
                df_practice_wide.to_csv(os.path.join(dest_dir, 'prac.csv'))
                df_test_wide.to_csv(os.path.join(dest_dir, 'test.csv'))
            if verbose:
                print('...PROCESS SUBJECT DATA [%s]' % (subject_id))

    # =================================================== #
    # CALCULATE AGG DATA
    # =================================================== #

    @staticmethod
    def calculate_agg_data(df, data_id):
        """
        df: df_model or df_subject
            df_model: read csv file from '...param_task1_actr0/markov-model1-sim-staydata.csv'
        :param data_id: param_id or subject_id
        :return: aggregate df (flatten columns)
               'pre_received_reward', 'state_frequency', 'state1_stay',
               'state1_stay_mean', 'state1_stay_std', 'state1_stay_sem',
               'state1_response_time_mean', 'state1_response_time_std',
               'state1_response_time_sem', 'state2_response_time_mean',
               'state2_response_time_std', 'state2_response_time_sem', 'data_id'],
        """
        # assert MaxLogLikelihood.MAXLL_FACTOR_VAR + ['state1_stay', 'state1_response_time', 'state2_response_time'] in df.columns
        group_var = MaxLogLikelihood.MAXLL_FACTOR_VAR
        dep_dict = {var:('mean', 'std', 'sem') for var in MaxLogLikelihood.MAXLL_DEP_VAR}
                    # {'state1_stay': ('mean', 'std', 'sem'),
                    #  'state1_response_time': ('mean', 'std', 'sem'),
                    #  'state2_response_time': ('mean', 'std', 'sem')}
        df_agg = df.groupby(group_var).\
            agg(dep_dict).reset_index()
        df_agg['data_id'] = data_id

        # flatten columns
        df_agg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in df_agg.columns]
        return df_agg

# def calculate_agg_data(data_path, idx):
#     """
#     Calculate the aggregated dataa cross epochs, index
#     :param data_path: the path of df
#     :param id: identification number, could be param_id or subject_id
#     :return: two aggregated dataframes, stay probability and response time by pre_received_reward and state_frequency
# =    """
#     assert (os.getcwd().split('/')[-1] == 'ACTR-MarkovTask') and (os.path.exists(data_path))
#
#     df = pd.read_csv(data_path)
#     dfstay = temporary_update_stay_probability(df)
#     dfstay = pd.merge(dfstay, df.drop(columns=['pre_received_reward']))
#
#     df_long = dfstay[['pre_received_reward', 'state_frequency', 'state1_response_time', 'state2_response_time']] \
#         .melt(id_vars=['pre_received_reward', 'state_frequency'],
#               value_vars=['state1_response_time', 'state2_response_time'],
#               var_name='state_name',
#               value_name='response_time')
#
#     df_long['state_name'] = df_long.apply(lambda x: x['state_name'].split('_')[0], axis=1)
#     df_long['state'] = df_long.apply(lambda x: x['state_name'] + ':' + x['state_frequency'], axis=1)
#     df_long = df_long.astype({'state': CategoricalDtype(categories=['state1:common',
#                                                                     'state1:rare',
#                                                                     'state2:common',
#                                                                     'state2:rare'], ordered=True)})
#     df_long = df_long.dropna()
#
#     dfstay_aggregate = dfstay \
#         .groupby(['pre_received_reward', 'state_frequency']) \
#         .agg(state1_stay_mean=('state1_stay', 'mean'),
#              state1_stay_sd=('state1_stay', 'std'),
#              state1_stay_se=('state1_stay', 'sem')).fillna(0.0).reset_index()
#
#     dfstay_aggregate['id'] = idx
#
#     dftime_aggregate = df_long.groupby(['pre_received_reward', 'state_name', 'state_frequency']) \
#         .agg(reponse_time_mean=('response_time', 'mean'),
#              reponse_time_sd=('response_time', 'std'),
#              reponse_time_se=('response_time', 'sem')).fillna(0.0).reset_index()
#
#     dftime_aggregate['id'] = idx
#     return dfstay_aggregate, dftime_aggregate

    # =================================================== #
    # SAVE AGG DATA
    # =================================================== #

    @staticmethod
    def save_agg_model_data(main_dir, model_dir, special_suffix='staydata', verbose=True, overwrite=False):
        """
        Save aggregated data to each simulated folder

        :param model_dir: 'data/param_simulation_test'
        :return:
        """
        # get all param_id
        param_ids = os.listdir(os.path.join(main_dir, model_dir))

        # save agg model to each param_id dir
        for param_id in param_ids:

            # get all model_paths contain *staydata.csv
            model_paths = glob.glob(os.path.join(main_dir, model_dir, param_id, '*%s.csv') % (special_suffix))

            for model_path in model_paths:
                # access model_name from path
                model_name =  'markov-' + model_path.split('/')[-1].split('-')[1]

                # calculate aggregate data
                df_model = pd.read_csv(model_path, index_col=0)

                # group variable
                df_model_agg = MaxLogLikelihood.calculate_agg_data(df=df_model, data_id=param_id)

                # define destination folder
                dest_dir = os.path.join(main_dir, model_dir, param_id, 'aggregate')
                # check exists
                if not Simulation.simple_check_exist(dest_dir=dest_dir, target_num_files=1, overwrite=overwrite, special_suffix='*-agg.csv'):
                    # save to aggregate dir
                    df_model_agg.to_csv('%s/%s-agg.csv' % (dest_dir, model_name), header=True, index=False)
                    if verbose:
                        print('\tSAVING MODEL[%s] AGG FILE...ID[%s]' % (model_name, param_id))


    @staticmethod
    def save_agg_subject_data(main_dir, subject_dir, overwrite=True, verbose=True):
        """
        Save aggregated data to each subject folder

        :param main_dir:
        :return:
        """

        # get all subject_ids
        subject_ids = os.listdir(os.path.join(main_dir, subject_dir))

        for subject_id in subject_ids:
            # get all subject_path contain *.csv: prac.csv and test.cav
            subject_paths = glob.glob(os.path.join(main_dir, subject_dir, subject_id, '*.csv'))

            for subject_path in subject_paths:

                # access trial type from path
                trial_type = subject_path.split('/')[-1].split('.')[0]

                # calculate aggregate data
                df_subject = pd.read_csv(subject_path, index_col=0)

                # group variable
                df_subject_agg = MaxLogLikelihood.calculate_agg_data(df=df_subject, data_id=subject_id)

                # define destination folder
                dest_dir = os.path.join(main_dir, subject_dir, subject_id, 'aggregate')
                # check exists
                if not Simulation.simple_check_exist(dest_dir=dest_dir,
                                                     target_num_files=2,
                                                     overwrite=overwrite,
                                                     special_suffix='*-agg.csv'):
                    # save to aggregate dir
                    df_subject_agg.to_csv(os.path.join(dest_dir, trial_type + '-agg.csv'), header=True, index=False)
                    if verbose: print('\tSAVING SUBJECT AGG FILE...ID [%s]' % (subject_id))

    # orin_dir = np.sort(glob.glob(subject_dir + '/[0-9]*'))
    #     for d in orin_dir:
    #         # destination folder
    #         dest_dir = '%s/%s' % (d, 'aggregate')
    #         # overwrite dir
    #         if overwrite:
    #             shutil.rmtree(dest_dir)
    #         if not os.path.exists(dest_dir):
    #             os.mkdir(dest_dir)
    #         for data_name in ['Pre3', 'Test3']:
    #             subject_id = d.split('/')[-1]
    #             orig_file = '%s/%s.csv' % (d, data_name)
    #             df_agg = MaxLogLikelihood.calculate_agg_data(orig_file, subject_id)
    #             df_agg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in df_agg.columns]
    #             df_agg.to_csv('%s/%s-agg.csv' % (dest_dir, data_name), header=True, index=False)
    #             print('\tSAVING SUBJECT AGG FILE...ID[%s] [%s]' % (subject_id, data_name))

    @staticmethod
    def load_model_agg_data(main_dir, model_dir, model_name):
        """
        load model agg data for all param_id and concat to one single df
        """
        model_agg_files = glob.glob(os.path.join(main_dir, model_dir, '*', 'aggregate', model_name + '-agg.csv'))
        model_agg = pd.concat([pd.read_csv(f).convert_dtypes() for f in model_agg_files], axis=0)
        model_agg['model_name'] = model_name
        return model_agg

    # =================================================== #
    # CALCULATE MAX LOGLIKELIHOOD
    # =================================================== #

    # def calculate_state1_stay_logprobz(subj_stay_agg, model_stay_agg, model_name):
    #     """
    #     Calculate z-normed log probability for 4 P(stay) data points
    #     :param subj_stay_agg:
    #     :param model_stay_agg:
    #     :param model_name:
    #     :return:
    #     """
    #     df_merge = pd.merge(subj_stay_agg, model_stay_agg, on=['pre_received_reward', 'state_frequency'],
    #                         suffixes=('.s', '.m'))
    #
    #     # calculate the LL(m | subject)
    #     df_merge['condition'] = df_merge.apply(lambda x: '%s|%s' % (x['pre_received_reward'], x['state_frequency']), axis=1)
    #     df_merge['state1_stay_z'] = df_merge.apply(
    #         lambda x: (x['state1_stay_mean.m'] - x['state1_stay_mean.s']) / max(x['state1_stay_sd.s'], 1e-10), axis=1)
    #     df_merge['state1_stay_probz'] = df_merge.apply(lambda x: norm.pdf(x['state1_stay_z']), axis=1)
    #     df_merge['state1_stay_logprobz'] = df_merge.apply(lambda x: np.log(max(x['state1_stay_probz'], 1e-10)),
    #                                                       axis=1)  # max(x['state1_stay_probz']), 1e-10)
    #     df_merge['model_name'] = model_name
    #     return df_merge
    #
    #
    # def calculate_rt_logprobz(subj_rt_agg, model_rt_agg, model_name):
    #     """
    #     Calculate z-normed log probability for 8 RTs data points
    #     :param subj_rt_agg:
    #     :param model_rt_agg:
    #     :param model_name:
    #     :return:
    #     """
    #     df_merge = pd.merge(subj_rt_agg, model_rt_agg, on=['pre_received_reward', 'state_frequency', 'state_name'],
    #                         suffixes=('.s', '.m'))
    #
    #     # calculate the LL(m | subject)
    #     df_merge['condition'] = df_merge.apply(
    #         lambda x: '%s|%s|%s' % (x['pre_received_reward'], x['state_frequency'], x['state_name']), axis=1)
    #     df_merge['rt_z'] = df_merge.apply(
    #         lambda x: (x['reponse_time_mean.m'] - x['reponse_time_mean.s']) / max(x['reponse_time_sd.s'], 1e-10), axis=1)
    #     df_merge['rt_probz'] = df_merge.apply(lambda x: norm.pdf(x['rt_z']), axis=1)
    #     df_merge['rt_logprobz'] = df_merge.apply(lambda x: np.log(max(x['rt_probz'], 1e-10)),
    #                                              axis=1)  # max(x['state1_stay_probz']), 1e-10)
    #     df_merge['model_name'] = model_name
    #     return df_merge
    #

    # def calculate_LL(subject_agg, model_agg, model_name):
    #     df_merge = pd.merge(subject_agg, model_agg, on=['pre_received_reward', 'state_frequency'], suffixes=('.s', '.m'))
    #     # state1stay
    #     df_merge[('state1_stay.s', 'z')] = df_merge.apply(
    #         lambda x: (x[('state1_stay.s', 'mean')] - x[('state1_stay.m', 'mean')]) / max(x[('state1_stay.s', 'std')],                                                                                  1e-10), axis=1)
    #     df_merge[('state1_stay.s', 'probz')] = df_merge.apply(lambda x: norm.pdf(x[('state1_stay.s', 'z')]), axis=1)
    #     df_merge[('state1_stay.s', 'logprobz')] = df_merge.apply(
    #         lambda x: np.log(max(x[('state1_stay.s', 'probz')], 1e-10)), axis=1)
    #
    #     # state1_response_time
    #     df_merge[('state1_response_time.s', 'z')] = df_merge.apply(
    #         lambda x: (x[('state1_response_time.s', 'mean')] - x[('state1_response_time.m', 'mean')]) / max(
    #             x[('state1_response_time.s', 'std')], 1e-10), axis=1)
    #     df_merge[('state1_response_time.s', 'probz')] = df_merge.apply(
    #         lambda x: norm.pdf(x[('state1_response_time.s', 'z')]), axis=1)
    #     df_merge[('state1_response_time.s', 'logprobz')] = df_merge.apply(
    #         lambda x: np.log(max(x[('state1_response_time.s', 'probz')], 1e-10)), axis=1)
    #
    #     # state2_response_time
    #     df_merge[('state2_response_time.s', 'z')] = df_merge.apply(
    #         lambda x: (x[('state2_response_time.s', 'mean')] - x[('state2_response_time.m', 'mean')]) / max(
    #             x[('state2_response_time.s', 'std')], 1e-10), axis=1)
    #     df_merge[('state2_response_time.s', 'probz')] = df_merge.apply(
    #         lambda x: norm.pdf(x[('state2_response_time.s', 'z')]), axis=1)
    #     df_merge[('state2_response_time.s', 'logprobz')] = df_merge.apply(
    #         lambda x: np.log(max(x[('state2_response_time.s', 'probz')], 1e-10)), axis=1)
    #
    #     # calculate sum of logprobz across 12 data points = loglikelihood
    #     df_merge['LL'] = df_merge[[(a, b) for (a, b) in df_merge.columns if b == 'logprobz']].sum().sum()
    #     df_merge['model_name'] = model_name
    #     return df_merge

    # =================================================== #
    # CALCULATE LL
    # =================================================== #

    @staticmethod
    def calculate_LL(subject_agg, model_agg, model_name):
        """
        Calculate LogLikelihood
        @param: subject_agg: one single subject aggregate data
        @param: model_agg: a combination of all param_id for one model (markov-model1) aggregate data
        """
        df_merge = pd.merge(subject_agg, model_agg, on=MaxLogLikelihood.MAXLL_FACTOR_VAR, suffixes=('.s', '.m'))

        group_var = MaxLogLikelihood.MAXLL_DEP_VAR
        for dv in group_var:
            # dv = ['state1_stay', 'state1_response_time', 'state2_response_time']
            # state1stay
            df_merge[dv+'_z'] = df_merge.apply(
                lambda x: (x[dv+'_mean.s'] - x[dv+'_mean.m']) / max(x[dv+'_std.s'], 1e-10), axis=1)
            df_merge[dv+'_probz'] = df_merge.apply(lambda x: norm.pdf(x[dv+'_z']), axis=1)
            df_merge[dv+'_logprobz'] = df_merge.apply(lambda x: np.log(max(x[dv+'_probz'], 1e-10)), axis=1)

        # calculate log-likelihood
        # need to group by 'data_id.m': parameter_id
        agg_dict = dict(zip([dv + '_logprobz' for dv in MaxLogLikelihood.MAXLL_DEP_VAR], [('sum')] * len(MaxLogLikelihood.MAXLL_DEP_VAR)))
        LL = pd.DataFrame(df_merge.groupby(['data_id.m']).agg(agg_dict).sum(axis=1), columns=['LL']).reset_index()
        df_LL = pd.merge(df_merge, LL, how='left', on='data_id.m')
        df_LL['model_name'] = model_name

        # reorder cols
        df_LL = MaxLogLikelihood.reorder_datafram_cols(df_LL)
        return df_LL



        # # rt1
        # df_merge['rt1_z'] = df_merge.apply(lambda x: (x['state1_response_time_mean.s'] - x['state1_response_time_mean.m']) / max(x['state1_response_time_std.s'], 1e-10), axis=1)
        # df_merge['rt1_probz'] = df_merge.apply(lambda x: norm.pdf(x['rt1_z']), axis=1)
        # df_merge['rt1_logprobz'] = df_merge.apply(lambda x: np.log(max(x['rt1_probz'], 1e-10)), axis=1)
        #
        # # rt2
        # df_merge['rt2_z'] = df_merge.apply(
        #     lambda x: (x['state2_response_time_mean.s'] - x['state2_response_time_mean.m']) / max(
        #         x['state2_response_time_mean.s'], 1e-10), axis=1)
        # df_merge['rt2_probz'] = df_merge.apply(lambda x: norm.pdf(x['rt2_z']), axis=1)
        # df_merge['rt2_logprobz'] = df_merge.apply(lambda x: np.log(max(x['rt2_probz'], 1e-10)), axis=1)

    @staticmethod
    def calculate_maxLL(df_LL, main_dir, model_dir):
        assert ('LL' in df_LL.columns)
        maxLL = df_LL['LL'].max()
        df_maxLL = df_LL[df_LL['LL'] == maxLL]

        # add parameter details
        res = MaxLogLikelihood.append_parameter_values(df_maxLL, main_dir=main_dir, model_dir=model_dir)

        # reorder cols
        res = MaxLogLikelihood.reorder_datafram_cols(res)
        return res

        # df_temp1 = dfLL_merged.groupby(['id.s', 'model_name']).agg(maxLL=('LL', 'max')).reset_index()
        # df_temp2 = pd.merge(dfLL_merged, df_temp1, how='left')
        # df_temp2['is_max_param_id'] = df_temp2.apply(lambda x: x['LL'] == x['maxLL'], axis=1)
        # df_temp2['param_value.m'] = df_temp2.apply(lambda x: MaxLogLikelihood.param_id2value(model_dir=model_dir, param_id=str(x['id.m'])), axis=1)
        # df_temp3 = df_temp2[df_temp2['is_max_param_id']].groupby(['id.s', 'model_name'])['id.m'].apply(list).reset_index(name='max_param_ids')
        # df_maxLL = pd.merge(df_temp2, df_temp3)
        # return df_maxLL


    @staticmethod
    def save_maxLL_data(df, main_dir, subject_dir, special_suffix='', overwrite=False, verbose=True):
        """
        Save LL data
        """
        assert (len(df['model_name'].unique()) == 1 and len(df['data_id.s'].unique()) == 1)

        df = df.reset_index().drop(columns=['index'])
        # define destination dir
        model_name = df['model_name'].unique()[0]
        subject_id = df['data_id.s'].unique()[0]
        dest_dir = os.path.join(main_dir, subject_dir, subject_id, 'maxll')

        dest_file_name = subject_id + '-' + model_name + special_suffix + '.csv'

        if not Simulation.simple_check_exist(dest_dir=dest_dir, target_num_files=2, overwrite=overwrite,
                                             verbose=verbose):
            df.to_csv(os.path.join(dest_dir, dest_file_name), index=False)

        return df

    @staticmethod
    def run_maxLL_pipline(main_dir, model_dir, raw_subject_dir, model_name, new_subject_dir: 'str', overwrite=False, verbose=True):
        """
        This function integrates all necessary steps to estimate maxLL
        """

        # step 0: pre-process subject data to match our model data structure
        MaxLogLikelihood.process_subject_data_online(main_dir=main_dir,
                                                     raw_subject_dir=raw_subject_dir,
                                                     new_subject_dir=new_subject_dir,
                                                     overwrite=overwrite,
                                                     verbose=verbose)

        # step 1: post-process subject data: aggregate by group variable
        MaxLogLikelihood.save_agg_subject_data(main_dir=main_dir,
                                               subject_dir=new_subject_dir,
                                               overwrite=overwrite,
                                               verbose=verbose)

        # step 2: post-process model data: aggregate by group variable
        MaxLogLikelihood.save_agg_model_data(main_dir=main_dir,
                                             model_dir=model_dir,
                                             special_suffix='staydata',
                                             overwrite=overwrite,
                                             verbose=verbose)

        # step 3: iterate subject data
        subject_ids = os.listdir(os.path.join(main_dir, new_subject_dir))
        for subject_id in subject_ids:
            for subject_data_type in ['prac', 'test']:
                subject_file = os.path.join(main_dir, new_subject_dir, subject_id, 'aggregate', subject_data_type+'-agg.csv')
                subject_agg = pd.read_csv(subject_file).convert_dtypes()

                # step 4: load model agg data for all param_id
                model_agg = MaxLogLikelihood.load_model_agg_data(main_dir=main_dir, model_dir=model_dir, model_name=model_name)

                # calcualte LL maxLL data
                df_LL = MaxLogLikelihood.calculate_LL(subject_agg, model_agg, model_name=model_name)
                df_maxLL = MaxLogLikelihood.calculate_maxLL(df_LL, main_dir=main_dir, model_dir=model_dir)

                # save LL maxLL data
                MaxLogLikelihood.save_maxLL_data(df_maxLL, main_dir=main_dir,
                                                 subject_dir=new_subject_dir,
                                                 special_suffix='-maxll',
                                                 overwrite=overwrite,
                                                 verbose=verbose)
                MaxLogLikelihood.save_maxLL_data(df_LL,
                                                 main_dir=main_dir,
                                                 subject_dir=new_subject_dir,
                                                 special_suffix='-ll',
                                                 overwrite=False,
                                                 verbose=verbose)

    # def calculate_maxLL(df1, df2, data_dir='data/model/param_simulation_0115'):
    #     """
    #     Calculate maxLL for each subject, each model
    #     12 datapoints are used: 4 P(stay) and 8 RTs
    #     :param df1:
    #     :param df2:
    #     :param data_dir:
    #     :return:
    #     """
    #     assert ('state1_stay_z' in df1.columns and 'rt_z' in df2.columns)
    #     # save LL
    #     df1_rename = df1.rename(
    #         columns={'state1_stay_z': 'z', 'state1_stay_probz': 'probz', 'state1_stay_logprobz': 'logprobz'})
    #     df2_rename = df2.rename(columns={'rt_z': 'z', 'rt_probz': 'probz', 'rt_logprobz': 'logprobz'})
    #
    #     df_merged = pd.concat([df1_rename, df2_rename], axis=0, ignore_index=True).groupby(
    #         ['id.s', 'id.m', 'model_name']).agg(LL=('logprobz', 'sum')).reset_index()
    #     df_temp1 = df_merged.groupby(['id.s', 'model_name']).agg(maxLL=('LL', 'max')).reset_index()
    #     df_temp2 = pd.merge(df_merged, df_temp1, how='left')
    #     df_temp2['is_max_param_id'] = df_temp2.apply(lambda x: x['LL'] == x['maxLL'], axis=1)
    #     # df_temp2['max_param_values'] = df_temp2.apply(lambda x: [param_id2value(data_dir=data_dir, param_id=param_id) for param_id in x], axis=1)
    #     df_temp2['param_value.m'] = df_temp2['id.m'].apply(lambda x: param_id2value(data_dir=data_dir, param_id=str(x)))
    #     df_temp3 = df_temp2[df_temp2['is_max_param_id']].groupby(['id.s', 'model_name'])['id.m'].apply(list).reset_index(
    #         name='max_param_ids')
    #     res = pd.merge(df_temp2, df_temp3)
    #     return res

    # @staticmethod
    # def save_max_loglikelihood_data(model_dir='data/model/param_simulation_0123',
    #                                 subject_dir='data/human/reformated_task_data',
    #                                 special_suffix='12dp', overwrite=False):
    #     """
    #
    #     :param model_dir:
    #     :param subject_dir:
    #     :param special_suffix:
    #     :param overwrite:
    #     :return:
    #     """
    #
    #     subject_ids = np.sort([i.split('/')[-1] for i in glob.glob('%s/[0-9]*' % (subject_dir))])
    #     models = np.sort([i.split('/')[-1] for i in glob.glob('%s/param_task0_actr0/aggregate/markov-model[0-9]-agg.csv' % (model_dir))])
    #     subj_data_types = np.sort([i.split('/')[-1] for i in glob.glob('%s/28326/aggregate/*-agg.csv' % (subject_dir))])
    #     param_ids = np.sort([i.split('/')[-1] for i in glob.glob('%s/param*' % (model_dir))])
    #
    #     simulation_id = model_dir.split('/')[-1]
    #
    #     for subject_id in subject_ids:
    #         for model_file in models:
    #             for subj_data_file in subj_data_types:
    #                 # each subject
    #                 subject_agg_file = os.path.join(subject_dir, subject_id, 'aggregate', subj_data_file)
    #                 subject_agg = pd.read_csv(subject_agg_file)
    #
    #                 dfLL_list = []
    #                 for param_id in param_ids:
    #                     # print(subject_id, model_file, subj_data_type, param_id)
    #                     dest_dir = os.path.join(subject_dir, subject_id, 'maxLL_' + simulation_id)
    #                     model_agg_file = os.path.join(model_dir, param_id, 'aggregate', model_file)
    #
    #                     if not os.path.exists(dest_dir):
    #                         os.mkdir(dest_dir)
    #
    #                     model_name = model_file.split('-')[1]
    #                     model_agg = pd.read_csv(model_agg_file)
    #                     dfLL = MaxLogLikelihood.calculate_LL(subject_agg, model_agg, model_name)
    #                     dfLL_list.append(dfLL)
    #
    #                 # save per subject per model
    #                 dfLL_merged = pd.concat(dfLL_list)
    #                 dfmaxLL = MaxLogLikelihood.calculate_maxLL(dfLL_merged, model_dir=model_dir)
    #
    #                 # incase if need special suffix
    #                 subj_data_type = str.lower(subj_data_file.split('-')[0])
    #                 dest_file_name = '%s/%s-%s-%sll-maxLL.csv' % (dest_dir, subj_data_type, model_name, special_suffix)
    #
    #                 # save file
    #                 if overwrite:
    #                     dfmaxLL.to_csv(dest_file_name, header=True, index=True)
    #                     print('SVING maxLL FILES...SUB-[%s] [%s] \tMODEL [%s]' % (subject_id, subj_data_type, model_name))
    #                 else:
    #                     if not os.path.exists(dest_file_name):
    #                         dfmaxLL.to_csv(dest_file_name, header=True, index=True)
    #                         print('SVING maxLL FILES...SUB-[%s] [%s] \tMODEL [%s]' % (subject_id, subj_data_type, model_name))
    #                     else:
    #                         print('SKIPPING...SUB-[%s]' % (subject_id))

        # =================================================== #
        # HELPER FUNCTIONS
        # =================================================== #

        # @staticmethod
        # def param_id2value(main_dir, model_dir='data/model/param_simulation_0115', param_id='param_task0_actr0',
        #                    return_dict=True):
        #     """
        #     fetch parameter values from data dir
        #     """
        #     assert (os.getcwd().split('/')[-1] == 'ACTR-MarkovTask')
        #     try:
        #         _, task_id, actr_id = param_id.split('_')
        #
        #         log_file = glob.glob('%s/param_%s_%s/log.csv' % (model_dir, task_id, actr_id))[0]
        #         log_param_dict = \
        #         pd.read_csv(log_file, header=0, index_col=0).drop(columns=['file_path']).drop_duplicates().to_dict(
        #             orient='records')[0]
        #         task_keys = ['M', 'RANDOM_WALK', 'REWARD']
        #         actr_keys = ['seed', 'ans', 'egs', 'alpha', 'lf', 'bll', 'mas']
        #
        #         task_dict = {key: log_param_dict[key] for key in task_keys}
        #         actr_dict = {key: log_param_dict[key] for key in actr_keys}
        #
        #         if return_dict:
        #             return task_dict, actr_dict
        #         else:
        #             # m, random_walk, r, seed, ans, egs, alpha, lf, bll, mas
        #             m, random_walk, r = task_dict.values()
        #             seed, ans, egs, alpha, lf, bll, mas = actr_dict.values()
        #             return m, random_walk, r, seed, ans, egs, alpha, lf, bll, mas
        #     except:
        # return None
    # =================================================== #
    # HELPER FUNCTIONS
    # =================================================== #

    @staticmethod
    def param_id2value(main_dir, model_dir, param_id, return_output='dict'):
        """
        Return a dict of parameter values based on parameter ID
        data_id.m provides parmeter id
        MaxLogLikelihood.MAXLL_PARAMETERS determines which parameters we decoded to use
        """
        assert return_output in ('dict', 'dataframe')
        try:
            log_file = os.path.join(main_dir, model_dir, param_id, 'log.csv')
            log = pd.read_csv(log_file, usecols=MaxLogLikelihood.MAXLL_PARAMETERS) \
                .drop_duplicates()  # .rename(columns={'REWARD': 'r', 'M': 'motivation'})
            log['REWARD'] = log.apply(lambda x: list(eval(x['REWARD']).values())[0], axis=1)
            log['data_id.m'] = param_id
            log_dict = log.to_dict(orient='records')[0]
            param_dict = {key: log_dict[key] for key in MaxLogLikelihood.MAXLL_PARAMETERS}

            if return_output == 'dict':
                return param_dict
            else:
                return log
        except:
            print('NO LOG FILE FOUND...[%s]' % (log_file))


    @staticmethod
    def append_parameter_values(df, main_dir, model_dir):
        """
        Apend parameter information if df contains param_id
        """
        assert ('data_id.m' in df.columns)
        res = pd.concat([df, df.apply(lambda x:
                                pd.Series(MaxLogLikelihood.param_id2value(
                                main_dir=main_dir,
                                model_dir=model_dir,
                                return_output='dict',
                                param_id=x['data_id.m'])),
                                axis=1)], axis=1)
        return res

    @staticmethod
    def reorder_datafram_cols(df):
        """
        Reorder columns
        """
        # reorder
        df = df.convert_dtypes()
        reorder_cols = list(df.select_dtypes(include='string').columns) + \
                       list(df.select_dtypes(exclude='string').columns)
        res = df[reorder_cols]
        return res


class Plot:
    REWARD_FACTOR = 'pre_received_reward'
    TRANS_FACTOR = 'pre_state_frequency'
    PLOT_FACTOR_VAR = [REWARD_FACTOR, TRANS_FACTOR]
    PALETTE = sns.color_palette(["#4374B3", "#FF0B04"])
    FIT_HEIGHT = 4
    FIG_WIDTH = 1.5 * FIT_HEIGHT

    @staticmethod
    def plot_response_switch(df, model_name, dep_var_suffix=''):
        """
        Plot state1_stay by pre_received_reward and pre_state_frequency
        :param df:
        :param model_name:
        :return:
        """
        assert set(Plot.PLOT_FACTOR_VAR + ['state1_stay' + dep_var_suffix]).issubset(set(df.columns))
        if len(dep_var_suffix) > 0:
            se = 'se' # enable se
        else:
            se = None

        fig, ax = plt.subplots(figsize=(Plot.FIG_WIDTH, Plot.FIT_HEIGHT))
        fig.suptitle('Summary: Stay Probability \n[%s]' % (model_name))

        sns.barplot(data=df, x=Plot.REWARD_FACTOR, y='state1_stay' + dep_var_suffix,
                    hue=Plot.TRANS_FACTOR, errorbar=se,
                    palette=Plot.PALETTE, alpha=.8,
                    order=['reward', 'non-reward'],
                    hue_order=['common', 'rare'],
                    ax=ax)

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='center')
        ax.axhline(0.5, color='grey', ls='-.', linewidth=.5)
        ax.set_ylim(0, 1.1)
        plt.show()

    @staticmethod
    def plot_response_switch_stripplot(df_agg, model_name, point_hue = 'epoch'):
        fig, axes = plt.subplots(1, 2, figsize=(1.5 * Plot.FIG_WIDTH, Plot.FIT_HEIGHT), sharex=True, sharey=True)
        fig.suptitle('Response Switch: [%s]' % (model_name))

        sns.stripplot(data=df_agg[df_agg[Plot.REWARD_FACTOR] == 'reward'], x=Plot.TRANS_FACTOR,
                      y='state1_stay_mean', s=5, marker="o", linewidth=0, alpha=.8, dodge=False,
                      order=['common', 'rare'], hue=Plot.TRANS_FACTOR, palette=Plot.PALETTE, ax=axes[0])
        sns.pointplot(data=df_agg[df_agg[Plot.REWARD_FACTOR] == 'reward'],
                      x=Plot.TRANS_FACTOR, y='state1_stay_mean',
                      order=['common', 'rare'], color='black', dodge=False, join=True, scale=2, errorbar='se',
                      markers='X', linestyles='solid', ax=axes[0])
        sns.pointplot(data=df_agg[df_agg[Plot.REWARD_FACTOR] == 'reward'],
                      x=Plot.TRANS_FACTOR, y='state1_stay_mean',
                      dodge=False, join=True,
                      order=['common', 'rare'], hue=point_hue, palette='gray',
                      markers='.', linestyles='dashed', ax=axes[0])

        sns.stripplot(data=df_agg[df_agg[Plot.REWARD_FACTOR] == 'non-reward'], x=Plot.TRANS_FACTOR,
                      y='state1_stay_mean', s=5, marker="o", linewidth=0, alpha=.8, dodge=False,
                      order=['common', 'rare'], hue=Plot.TRANS_FACTOR, palette=Plot.PALETTE, ax=axes[1])
        sns.pointplot(data=df_agg[df_agg[Plot.REWARD_FACTOR] == 'non-reward'],
                      x=Plot.TRANS_FACTOR, y='state1_stay_mean',
                      order=['common', 'rare'], color='black', dodge=False, join=True, scale=2, errorbar='se',
                      markers='X', linestyles='solid', ax=axes[1])
        sns.pointplot(data=df_agg[df_agg[Plot.REWARD_FACTOR] == 'non-reward'],
                      x=Plot.TRANS_FACTOR, y='state1_stay_mean',
                      dodge=False, join=True,
                      order=['common', 'rare'], hue=point_hue, palette='gray',
                      markers='.', linestyles='dashed', ax=axes[1])

        axes[0].set_ylim(0, 1.05)
        axes[0].set_title('Reward')
        axes[1].set_title('Non-Reward')
        axes[0].get_legend().remove()
        axes[1].get_legend().remove()

        plt.show()
    @staticmethod
    def plot_response_time(df, model_name, combine=True, dep_var_suffix=''):
        assert set(Plot.PLOT_FACTOR_VAR + ['state1_response_time' + dep_var_suffix, 'state2_response_time' + dep_var_suffix]).issubset(set(df.columns))
        df = df.copy()
        if df['state1_response_time' + dep_var_suffix].min() > 100:
            df['state1_response_time' + dep_var_suffix] = df['state1_response_time' + dep_var_suffix] /1000
            df['state2_response_time' + dep_var_suffix] = df['state2_response_time' + dep_var_suffix] / 1000
        df['response_time' + dep_var_suffix] = df['state1_response_time' + dep_var_suffix] + df['state2_response_time' + dep_var_suffix]

        if len(dep_var_suffix) > 0:
            se = 'se'  # enable se
        else:
            se = None
        if combine:
            fig, ax = plt.subplots(figsize=(Plot.FIG_WIDTH, Plot.FIT_HEIGHT))
            fig.suptitle('Summary: Response Time \n[%s]' % (model_name))
            sns.barplot(data=df, x=Plot.REWARD_FACTOR, y='response_time' + dep_var_suffix,
                        hue=Plot.TRANS_FACTOR, errorbar=se,
                        palette=Plot.PALETTE, alpha=.8,
                        order=['reward', 'non-reward'],
                        hue_order=['common', 'rare'],
                        ax=ax)

            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', label_type='center')
            ax.axhline(0.5, color='grey', ls='-.', linewidth=.5)
            ax.legend(loc='lower right')
            plt.show()
        else:
            fig, axes = plt.subplots(1, 2, figsize=(Plot.FIG_WIDTH * 2, Plot.FIT_HEIGHT))
            fig.suptitle('Summary: Response time: [%s]' % (model_name))
            sns.barplot(data=df, x=Plot.REWARD_FACTOR, y='state1_response_time' + dep_var_suffix,
                        hue=Plot.TRANS_FACTOR,
                        order=['reward', 'non-reward'],
                        hue_order=['common', 'rare'],
                        palette=Plot.PALETTE, alpha=1, ax=axes[0])
            sns.barplot(data=df, x=Plot.REWARD_FACTOR, y='state2_response_time' + dep_var_suffix,
                        hue=Plot.TRANS_FACTOR,
                        order=['reward', 'non-reward'],
                        hue_order=['common', 'rare'],
                        palette=Plot.PALETTE, alpha=.5, ax=axes[1])
            axes[0].set_title('state1')
            axes[1].set_title('state2')
            for ax in axes:
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', label_type='center')
            axes[1].get_legend().remove()
            plt.show()

    @staticmethod
    def plot_activation_trace(df1_atrace):
        assert set(['count', ':Reference-Count', ':Activation']).issubset(set(df1_atrace.columns))
        reward_colors = ['#F96666', '#FFD4D4'] * 4
        frequency_colors = ['#3C4048', '#B2B2B2', '#B2B2B2', '#3C4048']

        my_chunk_palette = sns.color_palette(frequency_colors + reward_colors)
        chunk_order = df1_atrace['memory'].sort_values().unique()

        fig, axes = plt.subplots(3, 1, figsize=(Plot.FIG_WIDTH, 4*Plot.FIT_HEIGHT), sharey=True, sharex=False)
        fig.suptitle('Summary: ACT-R Memory Trace')
        sns.barplot(data=df1_atrace, y='memory', x='count', order=chunk_order, palette=my_chunk_palette, ax=axes[0])
        sns.barplot(data=df1_atrace, y='memory', x=':Reference-Count', order=chunk_order, palette=my_chunk_palette,
                    ax=axes[1])
        sns.barplot(data=df1_atrace, y='memory', x=':Activation', order=chunk_order, palette=my_chunk_palette,
                    errorbar='se', ax=axes[2])

        axes[0].set_title('Experiment Trial Count')
        axes[1].set_title('ACTR Trace: :Retrieval-Count')
        axes[2].set_title('ACTR Trace: :Activation')
        plt.show()


    @staticmethod
    def plot_learning_performance(dfp, title):
        """
        Plot learning trajectory
        :param dfr: cumulative reward
        :param dfp: cumulative optimal response
        :return:
        """
        assert set(['optimal_response_sum_prop', 'received_reward_sum_prop']).issubset(set(dfp.columns))
        fig, ax = plt.subplots(figsize=(Plot.FIG_WIDTH, Plot.FIT_HEIGHT))
        fig.suptitle('Summary: [%s] Learning Performance (Cumulative)' %(title))
        sns.lineplot(data=dfp, x='index', y='optimal_response_sum_prop',
                     label='performance (cumulative)', color='steelblue', ax=ax)
        sns.lineplot(data=dfp, x='index', y='received_reward_sum_prop',
                     label='rewards (cumulative)', color='tomato', ax=ax)

        ax.axhline(0.5, color='grey', ls='-.', linewidth=.5)
        ax.set_ylim(0, 1)
        plt.show()
