Final project for DRL 8803 class :

On this branch, there is the code for offline training.


env_clean.py defined the 2d Hide and seek environment rewards etc

CQL is defined in agent.py

human_policy.py generates the expert trajectories

networks.py contains the network architecture

train.py train the networks

To train a shared policy for all agents on expert trajectory :
 python train.py --episodes=200 --env_size=10 --distinct_policy=False --n_walls=0 --run_name="Shared"

To train a policy for each team run :
 python train.py --episodes=200 --env_size=10 --distinct_policy=False --n_walls=0 --mixed=True --run_name="Disctint"  

To train a policy with a dataset with optimal and random trajectories run
 python train.py --episodes=200 --env_size=10 --distinct_policy=True --n_walls=0 --mixed=True --run_name="Mixed"  
 
