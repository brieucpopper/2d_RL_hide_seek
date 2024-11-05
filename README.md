Final project for DRL 8803 class


yml file enables creating a compatible python environment to run the project

env.py defined the 2d Hide and seek environment rewards etc


then proof_concept.py gives example simple policies and training
    For this one there are 3 main blocks :

    - RUN RANDOM POLICY AND PRINT RETURNS
    - TRAIN TWO PREDATORS WITH DQN TO CHASE RANDOM POLICY
    - TEST A SAVED WEIGHT FROM THE PREVIOUS TRAINING

Training should output something like

Episode: 0, Pred1 Reward: 361.24, Pred2 Reward: 375.11, Epsilon: 0.94

Episode: 5, Pred1 Reward: 419.12, Pred2 Reward: 413.63, Epsilon: 0.63

Episode: 10, Pred1 Reward: 442.86, Pred2 Reward: 440.47, Epsilon: 0.42

Episode: 15, Pred1 Reward: 456.68, Pred2 Reward: 432.30, Epsilon: 0.28
