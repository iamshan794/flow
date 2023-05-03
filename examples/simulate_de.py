"""Runner script for non-RL simulations in flow.

Usage
    python simulate.py EXP_CONFIG --no_render
"""
import argparse
import sys
import json
import os
from flow.core.experiment import Experiment

from flow.core.params import AimsunParams
from flow.utils.rllib import FlowParamsEncoder
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.registry import get_agent_class
from flow.utils.registry import make_create_env
from ray.rllib.agents.ppo import PPOAgent
import gym.envs.registration as reg
import gym.envs as envs 
import gym 


class newExperiment(Experiment):
    def __init__(self,flow_params,callables,env_name):
       super(newExperiment,self).__init__(flow_params=flow_params,custom_callables=callables) 
       #create_env,gym_name=make_create_env(flow_params)
       #register_env(env_name,create_env) 
       reg.register(env_name)
       print("************Registered model "+env_name+"***********")
             
       #self.env=create_env() 

       #print("LIST OF ENVS",list_envs())
def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        description="Parse argument used when running a Flow simulation.",
        epilog="python simulate.py EXP_CONFIG --num_runs INT --no_render")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/non_rl.')

    # optional input parameters
    # --------------------ENTER params.json path --------------------------------------------------
    parser.add_argument('--parameters',type=str,default='/root/ray_results/stabilizing_the_ring/PPOFinal/params.json')
    parser.add_argument(
        '--num_runs', type=int, default=1,
        help='Number of simulations to run. Defaults to 1.')
    parser.add_argument(
        '--no_render',
        action='store_true',
        help='Specifies whether to run the simulation during runtime.')
    parser.add_argument(
        '--aimsun',
        action='store_true',
        help='Specifies whether to run the simulation using the simulator '
             'Aimsun. If not specified, the simulator used is SUMO.')
    parser.add_argument(
        '--gen_emission',
        action='store_true',
        help='Specifies whether to generate an emission file from the '
             'simulation.')

    return parser.parse_known_args(args)[0]


if __name__ == "__main__":
    flags = parse_args(sys.argv[1:])

    # Get the flow_params object.
    module = __import__("exp_configs.rl.singleagent", fromlist=[flags.exp_config])
    flow_params = getattr(module, flags.exp_config).flow_params
    print(flow_params.keys(),flow_params.values())
    
    # Get the custom callables for the runner.
    if hasattr(getattr(module, flags.exp_config), "custom_callables"):
        callables = getattr(module, flags.exp_config).custom_callables
    else:
        callables = None

    flow_params['sim'].render = not flags.no_render
    flow_params['simulator'] = 'aimsun' if flags.aimsun else 'traci'

    # If Aimsun is being called, replace SumoParams with AimsunParams.
    if flags.aimsun:
        sim_params = AimsunParams()
        sim_params.__dict__.update(flow_params['sim'].__dict__)
        flow_params['sim'] = sim_params

    # Specify an emission path if they are meant to be generated.
    if flags.gen_emission:
        flow_params['sim'].emission_path = "./data"

        # Create the flow_params object
        fp_ = flow_params['exp_tag']
        dir_ = flow_params['sim'].emission_path
        with open(os.path.join(dir_, "{}.json".format(fp_)), 'w') as outfile:
            json.dump(flow_params, outfile,
                      cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # Create the experiment object.
    
    with open(flags.parameters) as json_file:
        config=json.load(json_file)
    print("config env is ",config['env'])

    #exp=newExperiment(flow_params=flow_params,callables=callables,env_name=config['env'])
    exp=Experiment(flow_params=flow_params,custom_callables=callables)
    ray.init()
    alg_name=config['env_config']['run']
    #agentcls=get_agent_class(alg_name)
      
    #agent=agentcls(config=config,env=config['env'])
    agent=PPOAgent(config=config,env=config['env'])
    #change checkpoints path --------------------------------------------------------- here 

    agent.restore('/root/ray_results/stabilizing_the_ring/PPOFinal/checkpoint_1140/checkpoint-1140')
    print("MODEL RESOTRED SUCCESSFULLY")
    #exp=newExperiment(flow_params=flow_params,callables=callables,env_name=config['env'])
    rl_actions=agent.compute_action # this is where we send the agent 
    # Run for the specified number of rollouts.
    ls=list(envs.registry.all())
    ls=[ls[i].id for i in range(len(ls))]
    print(str(config['env'] in ls),"VERIFIED REGISTRATION")
    gym.make(config['env'])


    exp.run(flags.num_runs, convert_to_csv=flags.gen_emission,rl_actions=rl_actions)
    
    
