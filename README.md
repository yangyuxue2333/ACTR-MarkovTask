# ACTR-MarkovTask

An ACT-R model of Markov-Two-Stage Task with motivation component implemented

## Markov Two-State Task

## Computational Modeling
[Demo](http://expfactory.org/experiments/two_stage_decision/preview)

This task assesses two types of reinforcement learning (RL): model-free and model-based RL. In this task, participants
make two sequential decisions that navigate them through two "stages" defined by different stimuli. First-stage choices
are associated with one of the two-second stages (e.g., 2a and 2b): one first-stage choice leads to 2a 70% of the time
and 2b 30% of the time, while the opposite is true of the other first-stage choice (i.e. 2a occurs 30% of the time and
2b occurs 70% of the time). Each second-stage choice is associated with some probability of receiving a reward. This
probability changes slowly over time, requiring continuous learning in order to succeed at the task. Because the goal of
the subject is to maximize rewards in the second stage, ideal performance would entail identifying the most rewarding
second stage (e.g., 2a) and making first-stage choices that make this result more likely (e.g., the first-stage choice
that results in 2a 70% of the time). 


![Two-State Task](https://docs.google.com/drawings/d/e/2PACX-1vSsR09U5a0U2w7mhSYHmaur7D0DxCN76zKHGHF9CsVq_weSO6J_1Mla1cSCWjQoNUS6Bxhp67GX3c3y/pub?w=317&h=216)


## Computational Model in ACT-R

### Model-Free

### Model-Base

### Motivation-Model

![Motivation-Model](https://docs.google.com/drawings/d/e/2PACX-1vTyGeadUXEmKtU6QbWB4gbKABAhI422PMzRjbwIbzmfrhSxv1BCLxI3hhKJtN2s0tg3p5MVKgvgVsrr/pub?w=613&h=854)

--- 

## Run Simulation

How to run model?

Note: need to start ACT-R standalone first

    from markov_simulate_test import * 
    
    # define task parameters
    r, m = 1, 1
    task_params = {'REWARD': {'B1': r, 'B2': r, 'C1': r, 'C2': r}, 'RANDOM_WALK': random_walk, 'M': m}

    # define actr parameters
    actr_params = {'ans': 0.5}
    
    # start simulation
    simulate_stay_probability(model="markov-model3", 
        epoch=1, n=300,
        task_params=task_params, 
        actr_params=actr_params, 
        log=None, 
        verbose=False)


## Data 

TBA