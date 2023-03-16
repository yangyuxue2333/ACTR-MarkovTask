import sys
import os

SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname('../__file__')), 'script')
sys.path.insert(0, SCRIPT_PATH)


from markov_pyactup import *
random.seed(0)

main_dir = os.path.dirname(os.getcwd())
subject_dir = os.path.join(main_dir, 'data', 'human', 'online_data')

# define output file dir
dest_dir = os.path.join(main_dir, 'data', 'model', 'param_optimization_test')
if not os.path.exists(dest_dir):
    print('CREATE...', dest_dir)
    os.mkdir(dest_dir)


# define subject and models
subject_ids = [str(i) for i in np.arange(1, 152)]
subject_ids = np.array_split(subject_ids, 5)[0] # divide into 5 split
estimate_models = ['markov-rl-mf', 'markov-rl-mb', 'markov-rl-hybrid', 'markov-ibl-mb', 'markov-ibl-hybrid']
epoch = 10


# start estimations
# start estimations
for i in tqdm(range(len(subject_ids))):
    subject_id = subject_ids[i]
    for estimate_model in estimate_models:
        for e in range(epoch):
            # if not os.path.exists(os.path.join(dest_dir, 'sub%s-%s-opt-result.csv' % (subject_id, estimate_model))):
            #     print('START SUB[%s] M[%s]' % (subject_id, estimate_model))
            MarkovEstimation.try_estimate(subject_id=subject_id, estimate_model=estimate_model, save_output=dest_dir, verbose=True)

print('...FINISHED...')