#!/usr/bin/env python3
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import sys
sys.path.append("..")


from simulation import run_irace_compatible_base_simulation

# Useful function to print errors.
def target_runner_error(msg):
    now = datetime.datetime.now()
    print(str(now) + " error: " + msg)
    sys.exit(1)



if __name__=='__main__':
    if len(sys.argv) < 5:
        print("\nUsage: ./target-runner.py <configuration_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
        sys.exit(1)

    # Get the parameters as command line arguments.
    configuration_id = sys.argv[1]
    instance_id = sys.argv[2]
    seed = sys.argv[3]
    instance = sys.argv[4]
    cand_params = sys.argv[5:]

    # Default values (if any)
    crossover_rate = None
    mutation_rate = None
    migration_pop_rate = None
    migration_interval = None
    # Parse parameters

    # Remove first element (instance name)
    cand_params.pop(0)    
    
    while cand_params:
        # Get and remove first and second elements.
        param = cand_params.pop(0)
        value = cand_params.pop(0)
        if param == "--cross_rate":
            crossover_rate = float(value)
        elif param == "--mut_rate":
            mutation_rate = float(value)
        elif param == "--mp_rate":
            migration_pop_rate = float(value)
        elif param == "--mig_inter":
            migration_interval = int(value)
        else:
            target_runner_error("unknown parameter %s" % (param))
    
    # Sanity checks
    if crossover_rate is None or mutation_rate is None or migration_pop_rate is None or migration_interval is None:
        target_runner_error("One or more parameters are missing. Please check the parameters.txt file.")
        sys.exit(1)
        
    
    # Run iRace compatible version of the simulation    
    result = run_irace_compatible_base_simulation(crossover_rate, mutation_rate, migration_pop_rate, migration_interval)
    print(result)
    
    sys.exit(0)


