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
	- days_count: number of days the data would be simulated
		starting from the last date available in "base_input_csv" file, the data will be simulated up to "days_count" number of days
	- simulation_count: the number of times the simulation will be run
		each time, the simulation will start from scratch
	- ts_update_day_count: basically the batch size, but day unit
		so, if "ts_update_day_count": 1, it means the parameters of Thompson Sampling will get updated every day
	
	- rand_sampling_applied: if random sampling wants to be applied in parallel
	- show_fig: figure will be plotted at the end if this parameter is set to True
	- NIG_priors: hyper parameters of Thompson sampling
		"NIG_priors":
        {
			"a": 1,
			"b": 1
        }

In, input csv file:
	- The columns of this file should match DIAMANTE or at least what you would like to deploy in DIAMANTE.
	- Column called "Participant", is the unique ID of the users for simulation.
		You can have multiple rows for a user which means there are different records for that user(record are stored per day basis)
		The joint value of the columns "Participant" and "Date" should be unique in the csv file
		The static features of a particular Participant should be the same in all rows available.
			e.g. Participant's gender should not be different in two different records
	- The features for each user should not be blank
	- There are columns associated with action variables and Reward. these columns can be left blank.
	- Those actions variables that are not blank will be used by the python code to train MAB model and to derive the values of a and b of
		Thompson sampling.
	- IMPORTANT: the latest date available in the csv file should be the same for all participants. For example:
		if Participant A has record for 2019-01-02 to 2019-01-06, Participant B should also have date 2019-01-06 in its record.
		Participant B can have records for any date prior to 2019-01-06, but cannot have records for after 2019-01-06.
		In this case "2019-01-06", should be the last date for all participants.
	- The MAB algorithm will find the the actions for each participant starting from the next day after the latest date 
		For example:
		in above example where last data was 2019-01-06, the algorithm start sending actions from date "2019-01-07" up to the number of days
		provided in json file. The data prior to "2019-01-06", will be only used to calculate "week_steps", "yesterday_steps" and etc. of the
		following date. For example, on "2019-01-07", the algorithm will look for record from "2019-01-01" to "2019-01-07" to calculate week_steps
		(if this data is available).







	
		
	
	
	
	
	