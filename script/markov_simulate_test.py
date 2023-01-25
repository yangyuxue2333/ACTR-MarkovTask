import shutil

import matplotlib.pyplot as plt
import seaborn as sns
from pandas import CategoricalDtype
from plotnine import *
from plotnine.data import *
from pathlib import Path
from tqdm import tqdm
import pandas.api.types as pdtypes
from markov_device import *
from datetime import date
import time
import glob
from scipy.stats import norm
import itertools

global convergence
convergence = 100


# =================================================== #
# SIMULATE
# =================================================== #
def simulate(model="markov-model1", n=20, task_params=None, actr_params=None, thresh=.6, verbose=False):
    """
    simulate markov model
    thresh determines whether the model learns optimal action sequence
    """
    global convergence
    if convergence < 0:
        print('>>> Failed to converge <<<')
        return
    m = MarkovACTR(setup=False)
    m.setup(model=model, verbose=verbose, task_params=task_params, actr_params=actr_params)
    m.run_experiment(n)
    df = m.df_postprocess_behaviors()

    perf = df['optimal_response_sum_prop'].loc[len(df) - 1]
    if perf >= thresh:
        if verbose: print(m)
        return m
    else:
        if verbose: print('>>> Not converged yet %.2f <<< [threshold = %.2f]' % (perf, thresh))
        convergence -= 1
        return simulate(model, n, task_params, actr_params, thresh)


def try_simulation_example(log_dir = '../data/model/param_simulation_test/', model_name="markov-model1", n=5, e=2):
    """
    in terminal
    >> ./script/
    >> python -c "from markov_simulate_test import *; try_simulation_example(log_dir = '../data/model/param_simulation_test/', model_name='markov-model3', n=500, e=50);"
    :param log_dir:
    :param model_name:
    :param n:
    :param e:
    :return:
    """
    # task parameter combination
    m = [0, 0.5, 1]  # [0, 0.5, 1, 1.5]
    r = [0.1, 5, 10]  # [3, 5, 10]
    random_walk = [True]

    # actr parameter combination
    ans = [0.2, 0.7]
    egs = [0.2, 0.7]
    alpha = [0.2, 0.7]
    lf = [0.5, 1]

    task_param_set = list(itertools.product(*[random_walk, m, r]))
    actr_param_set = list(itertools.product(*[ans, egs, alpha, lf]))

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
            if not check_parameters(log_file_path='%s%s%s' % (log_dir, param_folder_id, 'log.csv'),
                                    task_param_set=task_params,
                                    actr_param_set=actr_params,
                                    epoch=e,
                                    n=n,
                                    model_name=model_name,
                                    param_id=param_folder_id):
                simulate_stay_probability(model=model_name, epoch=e, n=n, task_params=task_params, actr_params=actr_params, log='%s%s' % (log_dir, param_folder_id))
                print("COMPLETE...%s" % (param_folder_id))
            else:
                print("SKIP ....%s" % (param_folder_id))

    print('RUNNING TIME: [%.2f]' % (time.time() - start_time))

# def simulate_response_monkey(model="markov-monkey", epoch=1, n=20):
#     rewards = 0.0
#     beh_list, state1stay_list = [], []
#
#     start_time = time.time()
#     for i in tqdm(range(epoch)):
#         m = MarkovHuman(model, verbose=False)
#         m.run_experiment(n)
#         if (i == 0): print(m)
#         m.run_experiment(n)
#
#         state1stay = m.calculate_stay_probability()
#         beh = m.df_postprocess_behaviors()
#         state1stay['epoch'] = i
#         beh['epoch'] = i
#
#         beh_list.append(beh)
#         state1stay_list.append(
#             # state1stay.groupby(['epoch', 'received_reward', 'reward_frequency', 'state_frequency']).agg(
#             state1stay.groupby(['epoch', 'received_reward', 'state_frequency']).agg(
#                 {'state1_stay': lambda x: x.mean(skipna=True)}).reset_index())
#
#         rewards += beh['received_reward'].sum() / len(beh)
#
#     max_id = np.argmax(list(m.task_parameters['REWARD_PROBABILITY'].values()))
#     expected_reward = list(m.task_parameters['REWARD'].values())[max_id] * \
#                       list(m.task_parameters['REWARD_PROBABILITY'].values())[max_id]
#     print('>>>>>>>>> SIMULATION REWARD GAINED <<<<<<<<<< \t EPOCH:', epoch)
#     print('GAINED R: %.2f (EXPECTED R: %.2f) \t [%.2f %%]\n\n\n\n' % (
#     (rewards / epoch), expected_reward, 100 * (rewards / epoch) / expected_reward))
#     df_beh = pd.concat(beh_list, axis=0)
#     df_state1stay = pd.concat(state1stay_list, axis=0)
#     return df_beh, df_state1stay


def simulate_stay_probability(model="markov-model1", epoch=1, n=20, task_params=None, actr_params=None, log=False, verbose=False):
    rewards = 0.0 
    beh_list, state1stay_list, utrace_list, atrace_list = [], [], [], []

    start_time = time.time()
    # for i in tqdm(range(epoch)):
    for i in range(epoch):
        m = simulate(model=model, n=n, task_params=task_params, actr_params=actr_params, thresh=0)
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
            file_path = log_simulation(beh, dir_path=log, file_name=model+'-sim-logdata')
            log_simulation(state1stay, dir_path=log, file_name=model+'-sim-staydata')
            log_simulation(utrace, dir_path=log, file_name=model+'-actr-udata')
            log_simulation(atrace, dir_path=log, file_name=model+'-actr-adata')
            
        else:
            beh_list.append(beh)
            state1stay_list.append(state1stay)
            utrace_list.append(utrace)
            atrace_list.append(atrace)

            # state1stay_list.append(state1stay.groupby(['epoch', 'm_parameter', 'received_reward', 'reward_frequency', 'state_frequency']).agg({'state1_stay': lambda x: x.mean(skipna=True)}).reset_index())
            #state1stay_list.append(state1stay.groupby(['epoch', 'pre_received_reward', 'state_frequency']).agg({'state1_stay': lambda x: x.mean(skipna=True)}).reset_index())
            #utrace_list.append(utrace.groupby(['epoch', 'action', 'state', 'response']).agg({':utility': lambda x: x.max(skipna=True)}).reset_index())
            #atrace_list.append(atrace.groupby(['epoch', 'memory', 'memory_type']).agg({':Reference-Count': lambda x: x.max(skipna=True), ':Activation': lambda x: x.max( skipna=True), ':Last-Retrieval-Activation': lambda x: x.max(skipna=True)}).reset_index())

            # plot
            rewards += beh['received_reward'].sum()/len(beh)

        
    if log:
        log_params(dir_path=log, epoch=epoch, n=n, actr_params=actr_params, task_params=task_params, file_path=file_path)
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

def log_simulation(df, dir_path='', file_name='', verbose=False):
    data_dir_path = '../data/'+dir_path
    if not os.path.exists(data_dir_path): 
        os.makedirs(data_dir_path)
    today = date.today().strftime('-%m%d%y') #.strftime('-%m-%d-%Y')
    # file_path=data_dir_path+file_name+today+'.csv'
    file_path = data_dir_path + file_name + '.csv'
    mode='w'
    header = True
    if os.path.exists(file_path):
        mode='a' 
        header = False
    df.to_csv(file_path, mode=mode, header=header)
    if verbose: print('>> saved..', file_name)
    return file_path

def log_params(dir_path, epoch, n, actr_params, task_params, file_path, verbose=False):
    param_dict={'epoch':epoch, 'n':n, **actr_params, **task_params,
                'model_name':file_path.split('/')[-1].split('-')[1][-1],
                'param_id':file_path.split('/')[-2],
                'file_path':file_path,}
    df = pd.DataFrame(param_dict.values(), index=param_dict.keys()).T
    
    data_dir_path = '../data/'+dir_path
    log_path =  data_dir_path+'log.csv'
    mode='w'
    header=True
    if os.path.exists(log_path):
        mode='a' 
        header=False
    df.to_csv(log_path, mode=mode, header=header, index=True)
    if verbose: print('>> saved..', log_path)
    

# =================================================== #
# LOAD SIMULATED DATA
# =================================================== #

def load_simulation(data_path='data/param_simulation_1114/param_task0_actr0', model_name='markov-model1', index_thres=None, verbose=True, load_agg=False):
    """
    Load simulated data for a single parameter set
    :param data_path:
    :param model_name: require full name of model e
    :param index_thres: 
    :param verbose:
    :return:
    """
    assert (os.getcwd().split("/")[-1] == 'ACTR-MarkovTask')

    df1_utrace = pd.read_csv(os.path.join(data_path, model_name + '-actr-udata.csv'))
    df1_utrace[':utility'] = df1_utrace[':utility'].apply(pd.to_numeric, errors='coerce')

    df1_atrace = pd.read_csv(os.path.join(data_path, model_name + '-actr-adata.csv'))
    df1_atrace[':Reference-Count'] = df1_atrace[':Reference-Count'].apply(pd.to_numeric, errors='coerce')
    df1_atrace[':Activation'] = df1_atrace[':Activation'].apply(pd.to_numeric, errors='coerce')
    df1_atrace[':Last-Retrieval-Activation'] = df1_atrace[':Last-Retrieval-Activation'].apply(pd.to_numeric,
                                                                                              errors='coerce')

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
            df1_state1stay = temporary_update_stay_probability(df1)

        if index_thres:
            df1 = df1[df1['index'] < index_thres]
            df1_state1stay = df1_state1stay[df1_state1stay['index'] < index_thres]
        return df1, df1_state1stay, df1_utrace, df1_atrace

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

def simple_check_exist(log_dir, param_folder_id, target_num_files):
    """
    Check whether simulation exist using the simplest way
    :param log_dir:
    :param param_folder_id:
    :return:
    """
    fs = glob.glob('../'+log_dir + param_folder_id + '/*.csv')
    print(fs)
    if len(fs) == target_num_files and 'log.csv' in fs:
        return True
    else:
        return False



# def get_df_log(log_dir='data/model/param_simulation_0121/'):
#     """
#     :return a df of all df_log contain simulation parameters
#     :param log_dir: 'data/model/param_simulation_0121/')
#     :return:
#     """
#     assert (os.getcwd().split("/")[-1] == 'ACTR-MarkovTask')
#     df_list = []
#     param_files = np.sort(glob.glob(log_dir + 'param*/log.csv'))
#     if len(param_files) == 0:
#         return pd.DataFrame()
# 
#     for f in param_files:
#         df_log = pd.read_csv(f, header=0, index_col=0).drop(columns=['file_path']).drop_duplicates()
#         df_log['param_id'] = f.split('/')[-2]
#         df_list.append(df_log)
# 
#     df_log = pd.concat(df_list, axis=0).reset_index()
#     df_log = df_log.assign(task_id=lambda x: x['param_id'].str.split('_', expand=True)[1].str.extract('(\d+)'),
#                            actr_id=lambda x: x['param_id'].str.split('_', expand=True)[2].str.extract('(\d+)')).astype({
#         'task_id': int, 'actr_id': int
#     })
#     return df_log
# 
# def check_parameters(log_dir, task_param_set, actr_param_set, epoch, n):
#     """
#     :param log_dir: 'data/model/param_simulation_0121/'
#     :param task_param_set:
#     :param actr_param_set:
#     :param epoch:
#     :param n:
#     :return:
#     """
#     '''
#     if not os.path.exists(log_file_path):
#         return False
#     log = pd.read_csv(log_file_path, header=0, index_col=0).drop(columns=['file_path'])
#     '''
#     log = get_df_log(log_dir=log_dir)
#     if log.empty:
#         return False
# 
#     log_param_list = log.to_records(index=False).tolist()
#     for log_param_set in log_param_list:
#         curr_param_set = (epoch, n,
#                           actr_param_set['seed'],
#                           actr_param_set['ans'],
#                           actr_param_set['egs'],
#                           actr_param_set['alpha'],
#                           # actr_param_set['v'],
#                           actr_param_set['lf'],
#                           actr_param_set['bll'],
#                           actr_param_set['mas'],
#                           str(task_param_set['REWARD']),
#                           task_param_set['RANDOM_WALK'],
#                           task_param_set['M'])
# 
#         if log_param_set == curr_param_set:
#             return True
#     return False
# 
# def get_next_param_id(log_dir, i, j):
#     """
#     :param log_dir: 'data/model/param_simulation_0121/'
#     :param i: 
#     :param j: 
#     :return: 
#     """
#     log = get_df_log(log_dir=log_dir)
#     if log.empty:
#         # print('no log')
#         next_task_id, next_actr_id = i, j
#     else:
#         # print('has log')
#         next_task_id = log['task_id'].max() + 1
#         next_actr_id = log['actr_id'].max() + 1
#     param_id = 'param_task%d_actr%d/' % (next_task_id, next_actr_id)
#     return param_id


def temporary_update_stay_probability(df):
    """
    Only use to fix 'pre_received_reward' problems on 0115 simulation
    Later simulated data have fixed this problem
    Calculate the probability of stay:
        A trial is marked as "STAY" if the agent selects the same action in current trial (e.g. LEFT)
        as the previous trial
    """
    dff = df.copy()
    dff['state1_stay'] = dff['state1_response'].shift() # first row is NA (look at previsou trial)
    dff['state1_stay'] = dff.apply(
        lambda x: 1 if x['state1_stay'] == x['state1_response'] else (np.nan if pd.isnull(x['state1_stay']) else 0),
        axis=1)
    dff['pre_received_reward'] = dff['received_reward'].shift()

    dff = dff.dropna(subset=['state1_stay', 'pre_received_reward'])
    dff.loc[:, ['pre_received_reward']] = dff.apply(lambda x: 'non-reward' if int(x['pre_received_reward']) == 0 else 'reward', axis=1)
    dff = dff.astype({'state_frequency': CategoricalDtype(categories=['common', 'rare'], ordered=True),
                    'pre_received_reward': CategoricalDtype(categories=['reward', 'non-reward'], ordered=True)})
    try:
        dff = dff[['epoch', 'index', 'state_frequency', 'received_reward', 'pre_received_reward', 'state1_stay', 'state1_response_time', 'state2_response_time']]
    except:
        dff['epoch'] = 0
        return dff
    return dff

def process_subject_data(data_dir='./data/human/task_data', log=None):
    assert (os.getcwd().split('/')[-1] == 'ACTR-MarkovTask')
    """
    Load and reformat emperical data (Teddy)
    """
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


# =================================================== #
# AGGREGATE SIMULATED DATA
# =================================================== #

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

def calculate_agg_data(data_path, param_id):
    """
    data_path ends with *staydata.csv
    :param data_path:
    :return:
    """
    df = pd.read_csv(data_path)
    if 'pre_received_reward' not in df.columns:
        df = temporary_update_stay_probability(df)
    df_agg = df.groupby(['state_frequency', 'pre_received_reward']).agg({'state1_stay': ('mean', 'std', 'sem'),
                                                                 'state1_response_time': ('mean', 'std', 'sem'),
                                                                 'state2_response_time': (
                                                                 'mean', 'std', 'sem')}).reset_index()
    df_agg['id'] = param_id
    return df_agg

def save_agg_model_data(model_dir = 'data/model/param_simulation_0115', overwrite=True):
    """
    Save aggregated data to each simulated folder

    :param main_dir:
    :return:
    """
    orin_dir = np.sort(glob.glob(model_dir + '/*'))

    for d in orin_dir:
        orig_files = np.sort(glob.glob('%s/*-sim-staydata.csv' % (d)))
        # destination folder
        dest_dir = '%s/%s' % (d, 'aggregate')
        # overwrite dir
        if overwrite:
            shutil.rmtree(dest_dir)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        for orig_file in orig_files:
            param_id = d.split('/')[-1]
            model_name = orig_file.split('-')[1][-1]
            df_agg = calculate_agg_data(orig_file, param_id)
            df_agg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in df_agg.columns]
            df_agg.to_csv('%s/markov-model%s-agg.csv' % (dest_dir, model_name), header=True, index=False)
            print('\tSAVING MODEL[%s] AGG FILE...ID[%s]' % (model_name, param_id))

def save_agg_subject_data(subject_dir = 'data/human/reformated_task_data', overwrite=True):
    """
    Save aggregated data to each subject folder

    :param main_dir:
    :return:
    """
    orin_dir = np.sort(glob.glob(subject_dir + '/[0-9]*'))
    for d in orin_dir:
        # destination folder
        dest_dir = '%s/%s' % (d, 'aggregate')
        # overwrite dir
        if overwrite:
            shutil.rmtree(dest_dir)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        for data_name in ['Pre3', 'Test3']:
            subject_id = d.split('/')[-1]
            orig_file = '%s/%s.csv' % (d, data_name)
            df_agg = calculate_agg_data(orig_file, subject_id)
            df_agg.columns = ['%s%s' % (a, '_%s' % b if b else '') for a, b in df_agg.columns]
            df_agg.to_csv('%s/%s-agg.csv' % (dest_dir, data_name), header=True, index=False)
            print('\tSAVING SUBJECT AGG FILE...ID[%s] [%s]' % (subject_id, data_name))

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
def calculate_LL(subject_agg, model_agg, model_name):
    df_merge = pd.merge(subject_agg, model_agg, on=['pre_received_reward', 'state_frequency'], suffixes=('.s', '.m'))
    # state1stay
    df_merge['state1_stay_z'] = df_merge.apply(
        lambda x: (x['state1_stay_mean.s'] - x['state1_stay_mean.m']) / max(x['state1_stay_mean.s'], 1e-10), axis=1)
    df_merge['state1_stay_probz'] = df_merge.apply(lambda x: norm.pdf(x['state1_stay_z']), axis=1)
    df_merge['state1_stay_logprobz'] = df_merge.apply(lambda x: np.log(max(x['state1_stay_probz'], 1e-10)), axis=1)

    # rt1
    df_merge['rt1_z'] = df_merge.apply(lambda x: (x['state1_response_time_mean.s'] - x['state1_response_time_mean.m']) / max(x['state1_response_time_mean.s'], 1e-10), axis=1)
    df_merge['rt1_probz'] = df_merge.apply(lambda x: norm.pdf(x['rt1_z']), axis=1)
    df_merge['rt1_logprobz'] = df_merge.apply(lambda x: np.log(max(x['rt1_probz'], 1e-10)), axis=1)

    # rt2
    df_merge['rt2_z'] = df_merge.apply(
        lambda x: (x['state2_response_time_mean.s'] - x['state2_response_time_mean.m']) / max(
            x['state2_response_time_mean.s'], 1e-10), axis=1)
    df_merge['rt2_probz'] = df_merge.apply(lambda x: norm.pdf(x['rt2_z']), axis=1)
    df_merge['rt2_logprobz'] = df_merge.apply(lambda x: np.log(max(x['rt2_probz'], 1e-10)), axis=1)

    # calculate log-likelihood
    df_merge['LL'] = df_merge[[c for c in df_merge.columns if '_logprobz' in c]].sum().sum()
    df_merge['model_name'] = model_name
    return df_merge
def param_id2value(model_dir='data/model/param_simulation_0115', param_id='param_task0_actr0', return_dict=True):
    """
    fetch parameter values from data dir
    """
    assert (os.getcwd().split('/')[-1] == 'ACTR-MarkovTask')
    try:
        _, task_id, actr_id = param_id.split('_')

        log_file = glob.glob('%s/param_%s_%s/log.csv' % (model_dir, task_id, actr_id))[0]
        log_param_dict = pd.read_csv(log_file, header=0, index_col=0).drop(columns=['file_path']).drop_duplicates().to_dict(
            orient='records')[0]
        task_keys = ['M', 'RANDOM_WALK', 'REWARD']
        actr_keys = ['seed', 'ans', 'egs', 'alpha', 'lf', 'bll', 'mas']

        task_dict = {key: log_param_dict[key] for key in task_keys}
        actr_dict = {key: log_param_dict[key] for key in actr_keys}

        if return_dict:
            return task_dict, actr_dict
        else:
            # m, random_walk, r, seed, ans, egs, alpha, lf, bll, mas
            m, random_walk, r = task_dict.values()
            seed, ans, egs, alpha, lf, bll, mas = actr_dict.values()
            return m, random_walk, r, seed, ans, egs, alpha, lf, bll, mas
    except:
        return None


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

def calculate_maxLL(dfLL_merged, model_dir='data/model/param_simulation_0115'):
    assert ('LL' in dfLL_merged.columns)
    df_temp1 = dfLL_merged.groupby(['id.s', 'model_name']).agg(maxLL=('LL', 'max')).reset_index()
    df_temp2 = pd.merge(dfLL_merged, df_temp1, how='left')
    df_temp2['is_max_param_id'] = df_temp2.apply(lambda x: x['LL'] == x['maxLL'], axis=1)
    df_temp2['param_value.m'] = df_temp2.apply(lambda x: param_id2value(model_dir=model_dir, param_id=str(x['id.m'])), axis=1)
    df_temp3 = df_temp2[df_temp2['is_max_param_id']].groupby(['id.s', 'model_name'])['id.m'].apply(list).reset_index(name='max_param_ids')
    df_maxLL = pd.merge(df_temp2, df_temp3)
    return df_maxLL



def save_max_loglikelihood_data(model_dir='data/model/param_simulation_0123',
                                subject_dir='data/human/reformated_task_data',
                                special_suffix='12dp', overwrite=False):
    """

    :param model_dir:
    :param subject_dir:
    :param special_suffix:
    :param overwrite:
    :return:
    """

    subject_ids = np.sort([i.split('/')[-1] for i in glob.glob('%s/[0-9]*' % (subject_dir))])
    models = np.sort([i.split('/')[-1] for i in glob.glob('%s/param_task0_actr0/aggregate/markov-model[0-9]-agg.csv' % (model_dir))])
    subj_data_types = np.sort([i.split('/')[-1] for i in glob.glob('%s/28326/aggregate/*-agg.csv' % (subject_dir))])
    param_ids = np.sort([i.split('/')[-1] for i in glob.glob('%s/param*' % (model_dir))])

    simulation_id = model_dir.split('/')[-1]

    for subject_id in subject_ids:
        for model_file in models:
            for subj_data_file in subj_data_types:
                # each subject
                subject_agg_file = os.path.join(subject_dir, subject_id, 'aggregate', subj_data_file)
                subject_agg = pd.read_csv(subject_agg_file)

                dfLL_list = []
                for param_id in param_ids:
                    # print(subject_id, model_file, subj_data_type, param_id)
                    dest_dir = os.path.join(subject_dir, subject_id, 'maxLL_' + simulation_id)
                    model_agg_file = os.path.join(model_dir, param_id, 'aggregate', model_file)

                    if not os.path.exists(dest_dir):
                        os.mkdir(dest_dir)

                    model_name = model_file.split('-')[1]
                    model_agg = pd.read_csv(model_agg_file)
                    dfLL = calculate_LL(subject_agg, model_agg, model_name)
                    dfLL_list.append(dfLL)

                # save per subject per model
                dfLL_merged = pd.concat(dfLL_list)
                dfmaxLL = calculate_maxLL(dfLL_merged, model_dir=model_dir)

                # incase if need special suffix
                subj_data_type = str.lower(subj_data_file.split('-')[0])
                dest_file_name = '%s/%s-%s-%sll-maxLL.csv' % (dest_dir, subj_data_type, model_name, special_suffix)

                # save file
                if overwrite:
                    dfmaxLL.to_csv(dest_file_name, header=True, index=True)
                    print('SVING maxLL FILES...SUB-[%s] [%s] \tMODEL [%s]' % (subject_id, subj_data_type, model_name))
                else:
                    if not os.path.exists(dest_file_name):
                        dfmaxLL.to_csv(dest_file_name, header=True, index=True)
                        print('SVING maxLL FILES...SUB-[%s] [%s] \tMODEL [%s]' % (subject_id, subj_data_type, model_name))
                    else:
                        print('SKIPPING...SUB-[%s]' % (subject_id))
