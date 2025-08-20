from copy import deepcopy
import datetime, os

from jmetal.operator import BinaryTournamentSelection
from jmetal.core.problem import BinaryProblem

from jmetal.operator.crossover import SBXCrossover, SPXCrossover
from jmetal.operator.mutation import SimpleRandomMutation, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from problems import ExpandedSchaffer, Griewank, Ackley

from algorithm import Runner
from algorithm.agents.base import BaseAgent
from analysis.constants_and_params import (
    OUTPUT_DIR,
    PROBLEMS_TO_TEST,
    MULTI_CLASS_SETUP,
    TRUST_MECHANISM,
    NUMBER_OF_RUNS,
    NUM_OF_VARS,
    CROSSOVER_RATE,
    MUTATION_RATE,
    MIGRATION,
    MIGRATION_POLICY,
    POPULATION_SIZE,
    OFFSPRING_POPULATION_SIZE,
    GENERATIONS_PER_SWAP,
    MAX_EVALUATIONS,
    AGENTS_NUMBER,
    STARTING_TRUST,
    NO_SEND_PENALTY,
    AUCTION_TRUST_WEIGHT,
    POPULATION_PART_TO_SWAP,
)

# Multi class setup parsing
agents = MULTI_CLASS_SETUP[0]
send_strategies = MULTI_CLASS_SETUP[1]
accept_strategies = MULTI_CLASS_SETUP[2]


def run_simulations_and_save_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now = datetime.datetime.now()
    start_date = (
        f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    )
    # migration_mechanism = "migration" if MIGRATION else "cloning"
    custom_output = f"{OUTPUT_DIR}/{TRUST_MECHANISM}_{MIGRATION_POLICY}_{start_date}"
   
    for i in range(NUMBER_OF_RUNS):
        now = datetime.datetime.now()
        exp_id = str(i+1)

        for problem in [problem_type(NUM_OF_VARS) for problem_type in PROBLEMS_TO_TEST]:
            
            # UNCOMMENT THE TYPE OF SIMULATION YOU WANT TO RUN
            
            """ MULTI AGENT CLASS SIMULATION """
            dir = f"{custom_output}/{problem.name()}"
            os.makedirs(dir, exist_ok=True)
            output_file_path = (
                f"{dir}/exp_{exp_id}.csv"
            )
            best_result = run_single_simulation(
                agents, problem, output_file_path, accept_strategies, send_strategies
            )
            print(
                f"Best result for {output_file_path}:",
                best_result,
            )
            
            """BASE AGENT CLASS SIMULATION"""
            # dir = f"{custom_output}/{problem.name()}"
            # os.makedirs(dir, exist_ok=True)
            # output_file_path = (
            #     f"{dir}/exp_{exp_id}.csv"
            # )
            # best_result = run_single_simulation(BaseAgent, problem, output_file_path, None, None)
            # print(
            #     f"Best result for {output_file_path}:",
            #     best_result,
            # )

            """SINGLE AGENT CLASS SIMULATION"""
            # for agent_class in AGENTS_TO_TEST:
            #     if agent_class is StrategyAgent:
            #         for accept_strategy in ACCEPT_STRATEGIES_TO_TEST:
            #             for send_strategy in SEND_STRATEGIES_TO_TEST:
            #                 output_file_path = f"{OUTPUT_DIR}/{agent_class.name()}_{accept_strategy}_{send_strategy}_{problem.name()}_{current_date}.csv"
            #                 run_single_simulation(
            #                     agent_class,
            #                     problem,
            #                     output_file_path,
            #                     accept_strategy,
            #                     send_strategy,
            #                 )
            #     else:
            #         output_file_path = f"{OUTPUT_DIR}/{agent_class.name()}_{problem.name()}_{current_date}.csv"
            #         run_single_simulation(
            #             agent_class, problem, output_file_path, None, None
            #         )


def run_single_simulation(
    agent_class,
    problem,
    output_file_path,
    accept_strategy,
    send_strategy,
    crossover_rate=CROSSOVER_RATE,
    mutation_rate=MUTATION_RATE,
    migration_pop_rate=POPULATION_PART_TO_SWAP,
    migration_interval=GENERATIONS_PER_SWAP,
    starting_trust=STARTING_TRUST,
    auction_weight=AUCTION_TRUST_WEIGHT,
    save_log=True,
):
    # print(f"{output_file_path=}")
    mutation = (
        BitFlipMutation(mutation_rate)
        if isinstance(problem, BinaryProblem)
        else SimpleRandomMutation(mutation_rate)
    )
    crossover = (
        SPXCrossover(crossover_rate) if isinstance(problem, BinaryProblem) else SBXCrossover(crossover_rate)
    )

    runner = Runner(
        output_file_path=output_file_path,
        agent_class=agent_class,
        agents_number=AGENTS_NUMBER,  # Needed only for single class, non strategy based agents
        generations_per_swap=migration_interval,
        problem=deepcopy(problem),
        population_size=POPULATION_SIZE,
        offspring_population_size=OFFSPRING_POPULATION_SIZE,
        mutation=mutation,
        crossover=crossover,
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=MAX_EVALUATIONS),
        send_strategy=send_strategy,
        accept_strategy=accept_strategy,
        migration=MIGRATION,
        trust_mechanism=TRUST_MECHANISM,
        starting_trust=starting_trust,
        no_send_penalty=NO_SEND_PENALTY,
        part_to_swap=migration_pop_rate,
        auction_weight=auction_weight,
        save_log=save_log,
    )
    runner.run_simulation()

    best_result = min(agent.algorithm.result().objectives[0] for agent in runner.agents)
    
    return best_result


def run_irace_compatible_base_simulation(crossover_rate, mutation_rate, migration_pop_rate, migration_interval, starting_trust, auction_weight):
    ''' Simulation for irace compatibility '''
    problem = Griewank(NUM_OF_VARS)  
    return run_single_simulation(agents, problem, "", accept_strategies, send_strategies, 
                                 crossover_rate=crossover_rate, mutation_rate=mutation_rate, migration_pop_rate=migration_pop_rate, 
                                 migration_interval=migration_interval, starting_trust=starting_trust, auction_weight=auction_weight, 
                                 save_log=False)


if __name__ == "__main__":
    run_simulations_and_save_results()