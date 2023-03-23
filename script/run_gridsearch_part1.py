import sys
import os
from datetime import date

SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname('../__file__')), 'script')
sys.path.insert(0, SCRIPT_PATH)


from markov_pyactup import *
random.seed(0)

TEST = False

# define dir
main_dir = os.path.dirname(os.getcwd())

# define subject and models
estimate_models = ['markov-ibl-hybrid']


dir_date = date.today().strftime("%m%y")

if TEST:
    estimate_models = ['markov-rl-hybrid', 'markov-ibl-hybrid']
    dir_date = 'test'


# define output file dir
dest_dir = os.path.join(main_dir, 'data', 'model', 'param_gs_%s' %(dir_date))
if not os.path.exists(dest_dir):
    print('CREATE...', dest_dir)
    os.mkdir(dest_dir)

# start estimations
for model_name in estimate_models:

    dest_dir = os.path.join(dest_dir, model_name)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        print('...CREATE ', dest_dir)

    est = MarkovEstimation(model_name=model_name)

    # prepare param log
    p_log = os.path.join(dest_dir, '%s-param-log.csv' % (model_name))
    if not os.path.exists(p_log):
        dfp = pd.DataFrame(est.param_gs_ls)
        dfp['model_name'] = est.kind
        dfp['param_id'] = ['param_id%05d' % i for i in dfp.index]
        dfp.to_csv(p_log, index=False)

    # start grid search simulation
    param_id_list = np.arange(len(est.param_gs_ls))
    param_id_list = np.array_split(param_id_list, 5)[0]  # divide into 5 split

    for param_id in param_id_list:
        MarkovEstimation.try_estimate_grid_search(dest_dir, model_name=model_name, param_id=param_id, verbose=True, overwrite=False)

print('...FINISHED...')