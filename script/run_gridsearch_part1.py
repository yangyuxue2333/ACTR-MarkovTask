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
estimate_models = ['markov-rl-hybrid', 'markov-ibl-hybrid']


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
    MarkovEstimation.try_estimate_grid_search(dest_dir, model_name=model_name, verbose=False)

print('...FINISHED...')