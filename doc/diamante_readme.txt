diamante.py is a linear contextual bandit algorithm tailored for DIAMANTE problem. We are using the same code base as we used for previous simulations, 
however there are some changes. This document explains the details of using this file for simulations.

If you want to use DIAMANTE realistic feature:

In JSON file, define:

	- noise parameter: std and mean
	- parameters of true model should
		name of parameters and their coefficients in a dictionary format. 
		the character "*" isreserved to show the interaction between variables.
		example:
		"true_coeff":
		{
			"intercept": 0,
			"age_year": 0,
			"action1": 3,
			"action1*age_year" : 2
		}
	- base_input_csv: path to where the base csv file is stored (explained later in this doc)
	- static_context_vars:
		list of all contextual variables that are unique for each user and does not change with time.
		NOTE: this element of the list should be avilable in the columns of base_input_csv file.
		exmaple:
		"static_context_vars": 
		[
			"gender_is_male",
			"gender_is_female",
			"health_status",
			"income_ladder",
			"age_year",
			"daily_goal",
			"weekly_goal"
		]
	- experiment_vars: the list of all action variables
		example:
		"experiment_vars": 
		[
			"action1",
			"action2"
		]
	- hypo_model_names: names of the hypo model you would like to train the model with
		You can define more than 1 models.
		The number of the provided names should match the number elements in "hypo_model_params" list.
		example:
		"hypo_model_names":
        [
			"hypo1",
			"hypo2"
		]
	- hypo_model_params: A list of list of hypothesized model parameters
		example:
		"hypo_model_params":
        [
			["intercept",
            "age_year",
            "action1"
			],
			["intercept",
            "age_year",
            "action1*age_year"
			]
		]
	- possible_actions: will be the arms of thomson sampling
		will be defined based on the values that each action variable can takes at each time
		example:
		"possible_actions": 
		[
			[0, 0],
			[0, 1],
			[1, 0],
			[1, 1]
		]
	- days_count:
		
		
	
	
	
	
	