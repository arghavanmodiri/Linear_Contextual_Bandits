# Linear_Contextual_Bandits
This repository is used for studying the linear contextual multi-arm bandit (MAB) algorithms. By generating simulation data based on
a desired distribution, one can study the effect of using MAB vs a standard A/B test. Currently, Thompson sampling and random sampling
have been implemented. Thompson Sampling has been chosen as it is one of the efficient algorithms in online decision problems. 

This repository is under development and new MAB algorithms will be added later.

## Getting Started
The file test.json can be used to adjust the parameters of the simulations (true model, hypo model, and 
the other required parameters of the simulations). 

### Defining the True and Hypo Model
Here is the list of the variables that specifies the design of the simulations.

- "true_model_params": Contains 
  - "true_coeff": the values of the coefficients of the true model
  - "noise": the mean and variance of the noise (to be included in the true model)
  - "context_vars": list of the names of the contextual variables
  - "experiment_vars": list of the names of the experimental variables

- dist_of_context: specifies the probability distribution of each contextual variables

- hypo_model_params: Contains the list of all the terms to be included in the hypo model

- possible_actions: All the possible values that all the experimental variables can possibly have at each time step

  If the experimental variables are defined as
  ```
  "experiment_vars": ["d1" , "d2"]
  ```
  Assuming d1 and d2 are binary variables, then possible_actions can be

  ```
  "possible_actions": [[0,0], [0,1], [1,0], [1,1]]
  ```

### Adjusting Other Parameters
The rest of the parameters such as number of users, batch_size, and etc. can be adjusted in test.json file.


## Running the Code
```
python main.py test.json
```
