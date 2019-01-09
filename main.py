def main(mode=None):
    """start the model"""


    #Step 1: Configuration (to be added in a seprate function)
    '''
    Parse the script command
    Check the csv files to make sure they are not empty
        True_Model_Coefficients.csv: will have true model coefficients
        Hypo_Model_Design.csv: will have the list of parameters to be used
    Create output folder if not existed and if in "Extensive Mode"
        Extensive Mode: details of the simulation will be saved in output files
        Test Mode: Just the rewards and basics will be printed on the screen
    Set the output files prefix. E.g. dec_01_1000users_
    and any other configuration to be set
    '''

    #Step 2: Setting the models
    '''
    Calls true_hypo_models.read_true_model to parse True_Model_Coefficients.csv
    Calls true_hypo_models.read_hypo_model to parse Hypo_Model_Design.csv
    Based on the variables, find the list of all bandit arms
    '''

    #Step 3: Calls the right policy
    '''
    Calls thompson sampling, random sampling, or any others which is specified
    by the user in command line (default: Calls thompson sampling)
        default priors: 0 unless it is specified by the user
    '''

    #Step 4: Plots
    '''
    Plots some basic figures. In "Extensive Mode", details will be saved so
    user can plots more figures if desired.
    '''



if __name__ == "__main__":
    main()