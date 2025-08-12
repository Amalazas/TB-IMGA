from algorithm.agents import StrategyAgent, AgentWithTrust
from algorithm.agents.base import BaseAgent
from algorithm.agents.strategy_based import AcceptStrategy, SendStrategy, TrustMechanism, MigrationPolicy
from problems import LABS, ExpandedSchaffer, Griewank, Ackley
from itertools import product

OUTPUT_DIR = "./test_output"

PLOTS_DIR = "./graphs"
BOX_AND_WHISKERS_PLOTS_DIR = f"{PLOTS_DIR}/box_and_whiskers"
MEAN_PLOTS_DIR = f"{PLOTS_DIR}/mean"
MULTI_CLASS_PLOTS_DIR = f"{PLOTS_DIR}/PERF_100vars_migration_local_trust"

SIGNIFICANCE_LEVEL = 0.05
NUMBER_OF_ITERATIONS = 998  # 9998 for 100000 evaluations, 998 for 10000 evaluations
ITERATION_INTERVAL = 50  # For plotting results only
TRUST_MECHANISM = TrustMechanism.Local
MIGRATION_POLICY = MigrationPolicy.TrustBasedAuction
NUMBER_OF_RUNS = 1
NUM_OF_VARS = 100
POPULATION_SIZE = 20
OFFSPRING_POPULATION_SIZE = 10
STARTING_TRUST = 10
RESTART_TRUST_THRESHOLD = 20 # 20 would be softcap on the worst possible trust, you can go below that # Should be a fairly difficult value to reach
GENERATIONS_PER_SWAP = 50
MAX_EVALUATIONS = 10000
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
MIGRATION = True
RESTARTING_ENABLED = False
AGENTS_NUMBER = 12 # For some reason, needs to be set to the final number of agents you've specified below
POPULATION_PART_TO_SWAP = 0.5
NO_SEND_PENALTY = int(POPULATION_SIZE * POPULATION_PART_TO_SWAP)
AUCTION_TRUST_WEIGHT = 0.5
AUCTION_SOLUTION_WEIGHT = 0.5

AGENTS_TO_TEST = [BaseAgent, StrategyAgent]
# TODO: add Ackley and other binary problem
PROBLEMS_TO_TEST = [
    Griewank,
    Ackley,
    ExpandedSchaffer,
]
ACCEPT_STRATEGIES_TO_TEST = [
    strategy
    for strategy in AcceptStrategy  # if strategy is not AcceptStrategy.Different
]
SEND_STRATEGIES_TO_TEST = [
    strategy for strategy in SendStrategy  # if strategy is not SendStrategy.Outlying
]

# TODO: get rid of this and use `AGENTS_TO_TEST` and `PROBLEMS_TO_TEST` directly.
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

# CUSTOM MULTI CLASS CONFIG (leave one config uncommented if you want to run it)
agents = []
send_strategies = []
accept_strategies = []

""" 5Creative_3Trust_1Perf_1Solo """
# for _ in range(3):  # Trust Agents
#     agents.append(AgentWithTrust)
#     send_strategies.append(None)
#     accept_strategies.append(None)
# for _ in range(1):  # Solo Agent
#     agents.append(StrategyAgent)
#     send_strategies.append(SendStrategy.Dont)
#     accept_strategies.append(AcceptStrategy.Reject)
# for _ in range(5):  # Creative Agents
#     agents.append(StrategyAgent)
#     send_strategies.append(SendStrategy.Outlying)
#     accept_strategies.append(AcceptStrategy.Different)
# for _ in range(1):  # Perfectionist Agents
#     agents.append(StrategyAgent)
#     send_strategies.append(SendStrategy.Best)
#     accept_strategies.append(AcceptStrategy.Better)

""" 1Extractor_2Tryhard_2Filter_6Creative """
# for _ in range(1):
#     agents.append(StrategyAgent)  # Extractor
#     send_strategies.append(SendStrategy.Dont)
#     accept_strategies.append(AcceptStrategy.Better)
# for _ in range(2):
#     agents.append(StrategyAgent)  # Tryhard
#     send_strategies.append(SendStrategy.Best)
#     accept_strategies.append(AcceptStrategy.Always)
# for _ in range(2):
#     agents.append(StrategyAgent)  # Filter
#     send_strategies.append(SendStrategy.Best)
#     accept_strategies.append(AcceptStrategy.Different)
# for _ in range(6):
#     agents.append(StrategyAgent)  # Creative
#     send_strategies.append(SendStrategy.Outlying)
#     accept_strategies.append(AcceptStrategy.Different)

"""All different mixes."""
# for send_strategy in SendStrategy:
#     for accept_strategy in AcceptStrategy:
#         agents.append(StrategyAgent)
#         send_strategies.append(send_strategy)
#         accept_strategies.append(accept_strategy)

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
