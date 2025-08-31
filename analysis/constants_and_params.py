from algorithm.agents import StrategyAgent, AgentWithTrust
from algorithm.agents.base import BaseAgent
from algorithm.agents.strategy_based import AcceptStrategy, SendStrategy, TrustMechanism, MigrationPolicy
from problems import LABS, ExpandedSchaffer, Griewank, Ackley
from itertools import product


SIGNIFICANCE_LEVEL = 0.05 # STATISTICAL TEST PARAMETER


### PLOT PARAMETERS ############################################################
PLOTS_DIR = "./final_graphs"
BOX_AND_WHISKERS_PLOTS_DIR = f"{PLOTS_DIR}/box_and_whiskers"
MEAN_PLOTS_DIR = f"{PLOTS_DIR}/mean"
MULTI_CLASS_PLOTS_DIR = f"{PLOTS_DIR}/multi_class"
NUMBER_OF_ITERATIONS = 19998  # USED AS HIGH BOUND FOR PLOTTING SCRIPTS # 9998 for 100000 evaluations, 998 for 10000 evaluations
ITERATION_INTERVAL = 50  # resolution of x axis in plots
################################################################################


### SIMULATION PARAMETERS ######################################################
# Experiment Type
TRUST_MECHANISM = TrustMechanism.Global
MIGRATION_POLICY = MigrationPolicy.TrustBasedAuction
STARTING_TRUST = 12 # Lower trust value means higher trust level (I know it's confusing, sry about that)
AUCTION_TRUST_WEIGHT = 0.4
# Base Experiment Parameters
OUTPUT_DIR = "./final_exp_output"
MAX_EVALUATIONS = 20000 # STOPPING CRITERION, COUNTED PER AGENT
NUMBER_OF_RUNS = 1
POPULATION_SIZE = 20
OFFSPRING_POPULATION_SIZE = 10
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
AGENTS_NUMBER = 12
MIGRATION = True
GENERATIONS_PER_SWAP = 50
POPULATION_PART_TO_SWAP = 0.5
NUM_OF_VARS = 100
PROBLEMS_TO_TEST = [
    Griewank,
    Ackley,
    ExpandedSchaffer,
]
###########################
# CUSTOM MULTI CLASS CONFIG
agents = []
send_strategies = []
accept_strategies = []

""" 3 per type - reasonable ones"""
for (send_strategy, accept_strategy) in product(
    [SendStrategy.Outlying, SendStrategy.Best],
    [AcceptStrategy.Better, AcceptStrategy.Different],
):
    for _ in range(3):
        agents.append(StrategyAgent)
        send_strategies.append(send_strategy)
        accept_strategies.append(accept_strategy)

MULTI_CLASS_SETUP = [agents, send_strategies, accept_strategies]
###################################
### Single agent simulations CONFIG
AGENTS_TO_TEST = [BaseAgent, StrategyAgent] 
ACCEPT_STRATEGIES_TO_TEST = [
    strategy
    for strategy in AcceptStrategy  # if strategy is not AcceptStrategy.Different
]
SEND_STRATEGIES_TO_TEST = [
    strategy for strategy in SendStrategy  # if strategy is not SendStrategy.Outlying
]
################################################################################


####################################################
### NOT AVAILABLE - CURRENTLY DISCARDED FEATURES ###
RESTARTING_ENABLED = False
RESTART_TRUST_THRESHOLD = 20 # 20 would be softcap on the worst possible trust, you can go below that # Should be a fairly difficult value to reach
NO_SEND_PENALTY = int(POPULATION_SIZE * POPULATION_PART_TO_SWAP)
####################################################


###################################################################
### OLD EXPERIMENTS AND PLOTTING SETUP - LEFT FOR COMPATIBILITY ###
# Experiment names order matters!!!
# It's used later for plotting order.
# Group the names by problem type and have
# the order of agents consistent between
# the problem types.
EXPERIMENTS = []
# for problem in PROBLEMS_TO_TEST:
#     for agent in AGENTS_TO_TEST:
#         if agent is StrategyAgent:
#             for accept_strategy in ACCEPT_STRATEGIES_TO_TEST:
#                 for send_strategy in SEND_STRATEGIES_TO_TEST:
#                     EXPERIMENTS.append(
#                         f"{agent.name()}_{accept_strategy}_{send_strategy}_{problem.name()}"
#                     )
#         else:
#             EXPERIMENTS.append(f"{agent.name()}_{problem.name()}")
