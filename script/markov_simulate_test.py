

import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from plotnine.data import *
from pathlib import Path
from tqdm import tqdm
import pandas.api.types as pdtypes
from markov_device import *
from datetime import date
import time

global convergence
convergence = 100


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


def try_simulation_example():
    r, r1, r2 = 2, 5, 10
    n = 50
    thresh = 0.6
    model1 = "markov-model1"
    model2 = "markov-model2"
    model3 = "markov-model3"

    actr_params = {'seed': '[100, 0]', 'v': 'nil', 'ans': 0.2, 'egs': 0.2, 'lf': 0.5, 'bll': 0.5}
    task_params = {'REWARD': {'B1': r, 'B2': r, 'C1': r, 'C2': r}}
    task_params1 = {'REWARD': {'B1': r1, 'B2': r1, 'C1': r1, 'C2': r1}}
    task_params2 = {'REWARD': {'B1': r2, 'B2': r2, 'C1': r2, 'C2': r2}}

    m1 = simulate(model1, n=n, task_params=task_params, actr_params=actr_params, thresh=thresh)
    m2 = simulate(model2, n=n, task_params=task_params, actr_params=actr_params, thresh=thresh)
    m3 = simulate(model3, n=n, task_params=task_params, actr_params=actr_params, thresh=thresh)
    return m1, m2, m3


def simulate_response_monkey(model="markov-monkey", epoch=1, n=20):
    rewards = 0.0
    beh_list, state1stay_list = [], []

    start_time = time.time()
    for i in tqdm(range(epoch)):
        m = MarkovHuman(model, verbose=False)
        m.run_experiment(n)
        if (i == 0): print(m)
        m.run_experiment(n)

        state1stay = m.calculate_stay_probability()
        beh = m.df_postprocess_behaviors()
        state1stay['epoch'] = i
        beh['epoch'] = i

        beh_list.append(beh)
        state1stay_list.append(
            # state1stay.groupby(['epoch', 'received_reward', 'reward_frequency', 'state_frequency']).agg(
            state1stay.groupby(['epoch', 'received_reward', 'state_frequency']).agg(
                {'state1_stay': lambda x: x.mean(skipna=True)}).reset_index())

        rewards += beh['received_reward'].sum() / len(beh)

    max_id = np.argmax(list(m.task_parameters['REWARD_PROBABILITY'].values()))
    expected_reward = list(m.task_parameters['REWARD'].values())[max_id] * \
                      list(m.task_parameters['REWARD_PROBABILITY'].values())[max_id]
    print('>>>>>>>>> SIMULATION REWARD GAINED <<<<<<<<<< \t EPOCH:', epoch)
    print('GAINED R: %.2f (EXPECTED R: %.2f) \t [%.2f %%]\n\n\n\n' % (
    (rewards / epoch), expected_reward, 100 * (rewards / epoch) / expected_reward))
    df_beh = pd.concat(beh_list, axis=0)
    df_state1stay = pd.concat(state1stay_list, axis=0)
    return df_beh, df_state1stay


def simulate_stay_probability(model="markov-model1", epoch=1, n=20, task_params=None, actr_params=None, log=False):
    rewards = 0.0 
    beh_list, state1stay_list, utrace_list, atrace_list = [], [], [], []

    start_time = time.time()
    for i in tqdm(range(epoch)):
        m = simulate(model=model, n=n, task_params=task_params, actr_params=actr_params, thresh=0)
        if (i==0): print(m)

        # stay probability
        state1stay = m.calculate_stay_probability()
        utrace, atrace = m.df_postprocess_actr_traces()
        beh = m.df_postprocess_behaviors()
        state1stay['epoch'] = i
        utrace['epoch'] = i
        atrace['epoch'] = i
        beh['epoch'] = i
        
        if log:
            df_beh, df_state1stay, df_utrace, df_atrace = beh, state1stay, utrace, atrace
            file_path = log_simulation(beh, dir_path=log, file_name=model+'-sim-logdata')
            log_simulation(state1stay, dir_path=log, file_name=model+'-sim-staydata')
            log_simulation(utrace, dir_path=log, file_name=model+'-actr-udata')
            log_simulation(atrace, dir_path=log, file_name=model+'-actr-adata')
            
        else:
            beh_list.append(beh)
            # state1stay_list.append(state1stay.groupby(['epoch', 'm_parameter', 'received_reward', 'reward_frequency', 'state_frequency']).agg({'state1_stay': lambda x: x.mean(skipna=True)}).reset_index())
            state1stay_list.append(state1stay.groupby(['epoch', 'm_parameter', 'received_reward', 'state_frequency']).agg({'state1_stay': lambda x: x.mean(skipna=True)}).reset_index())
            utrace_list.append(utrace.groupby(['epoch', 'index', 'action', 'state', 'response']).agg({':utility': lambda x: x.max(skipna=True)}).reset_index())
            atrace_list.append(atrace.groupby(['epoch', 'index', 'memory', 'memory_type']).agg({':Reference-Count': lambda x: x.max(skipna=True),
                                                                                                ':Activation': lambda x: x.max( skipna=True),
                                                                                                ':Last-Retrieval-Activation': lambda x: x.max(skipna=True)}).reset_index())

            # state1stay_list.append(state1stay)
            # utrace_list.append(utrace)
            # atrace_list.append(atrace)

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
    param_dict={'epoch':epoch, 'n':n, **actr_params, **task_params, 'file_path':file_path}
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
    

def load_simulation(data_path='data/param_simulation_1114/param_id0', model_name='markov-model1', index_thres=None, verbose=True):
    assert (os.getcwd().split("/")[-1] == 'ACTR-MarkovTask')
    df1 = pd.read_csv(os.path.join(data_path, model_name + '-sim-logdata.csv')).dropna(axis=0).apply(pd.to_numeric,
                                                                                                     errors='ignore')
    df1_state1stay = pd.read_csv(os.path.join(data_path, model_name + '-sim-staydata.csv')).dropna(axis=0).apply(
        pd.to_numeric, errors='ignore')

    if index_thres:
        df1 = df1[df1['index'] < index_thres]
        df1_state1stay = df1_state1stay[df1_state1stay['index'] < index_thres]

    df1_utrace = pd.read_csv(os.path.join(data_path, model_name + '-actr-udata.csv'))
    df1_utrace[':utility'] = df1_utrace[':utility'].apply(pd.to_numeric, errors='coerce')

    df1_atrace = pd.read_csv(os.path.join(data_path, model_name + '-actr-adata.csv'))
    df1_atrace[':Reference-Count'] = df1_atrace[':Reference-Count'].apply(pd.to_numeric,errors='coerce')
    df1_atrace[':Activation'] = df1_atrace[':Activation'].apply(pd.to_numeric, errors='coerce')
    df1_atrace[':Last-Retrieval-Activation'] = df1_atrace[':Last-Retrieval-Activation'].apply(pd.to_numeric, errors='coerce')

    param_log = pd.read_csv(os.path.join(data_path, 'log.csv')).loc[0]

    if verbose:
        print('...SUCCESSFULLY LOADED DATA...')
        print(param_log)
    return df1, df1_state1stay, df1_utrace, df1_atrace