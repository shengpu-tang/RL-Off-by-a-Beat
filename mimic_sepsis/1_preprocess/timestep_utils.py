import numpy as np
import pandas as pd

S_survival = 750
S_death = 751

# trajectory of (s,a,r,s',a'), original timestep
def make_traj_orig(df_in):
    df = df_in.copy()
    
    # populate the next state and next action
    df['s:next_state'] = df['s:state'].shift(-1)
    df['a:next_action'] = df['a:action'].shift(-1)

    # add terminal state and done flag
    df.loc[df['r:reward'] == 1, 's:next_state'] = S_survival
    df.loc[df['r:reward'] == -1, 's:next_state'] = S_death
    df['done'] = (df['r:reward'] != 0).astype(int)
    
    # restore integer data types
    df[['s:state', 'a:action']] = df[['s:state', 'a:action']].astype('Int64')
    df[['s:next_state', 'a:next_action']] = df[['s:next_state', 'a:next_action']].astype('Int64')
    
    return df[[
        'step', 's:state', 'a:action', 'r:reward', 's:next_state', 'a:next_action', 'done'
    ]]


# trajectory of (s,a,r,s',a'), original timestep but dropping last step
def make_traj_dropped(df_in):
    df = df_in.copy()
    
    # populate the next state and next action
    df['s:next_state'] = df['s:state'].shift(-1)
    df['a:next_action'] = df['a:action'].shift(-1)

    # shift last step reward to one step earlier so we keep it after dropping the last step, only works since this is a sparse reward setting
    df['r:reward'] = df['r:reward'].shift(-1) 
    
    # drop last step
    df = df.iloc[:-1]

    # add terminal state and done flag
    df.loc[df['r:reward'] == 1, 's:next_state'] = S_survival
    df.loc[df['r:reward'] == -1, 's:next_state'] = S_death
    df['done'] = (df['r:reward'] != 0).astype(int)

    # restore integer data types
    df[['s:state', 'a:action']] = df[['s:state', 'a:action']].astype('Int64')
    df[['s:next_state', 'a:next_action']] = df[['s:next_state', 'a:next_action']].astype('Int64')
    
    return df[[
        'step', 's:state', 'a:action', 'r:reward', 's:next_state', 'a:next_action', 'done'
    ]]


# trajectories (s,a,r,s',a') for shifted timestep
def make_traj_shifted(df_in):
    df = df_in.copy()

    # Shift action time index by -1: move the next action backwards so match with current state
    # step 0 action is removed
    # step T action is NaN
    df['a:curr_action'] = df['a:action']
    df['a:action'] = df['a:curr_action'].shift(-1)

    # populate the next state and next action
    df['s:next_state'] = df['s:state'].shift(-1)
    df['a:next_action'] = df['a:action'].shift(-1)

    # shift last step reward to one step earlier so we keep it after dropping the last step, only works since this is a sparse reward setting
    df['r:reward'] = df['r:reward'].shift(-1)

    # drop last step (with NaN action)
    df = df.iloc[:-1]

    # add terminal state and done flag
    df.loc[df['r:reward'] == 1, 's:next_state'] = S_survival
    df.loc[df['r:reward'] == -1, 's:next_state'] = S_death
    df['done'] = (df['r:reward'] != 0).astype(int)
    
    # restore integer data types
    df[['s:state', 'a:action']] = df[['s:state', 'a:action']].astype('Int64')
    df[['s:next_state', 'a:next_action']] = df[['s:next_state', 'a:next_action']].astype('Int64')
    
    return df[[
        'step', 's:state', 'a:action', 'r:reward', 's:next_state', 'a:next_action', 'done'
    ]]
