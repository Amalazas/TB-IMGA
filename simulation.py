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
    MIGRATION,
    POPULATION_SIZE,
    OFFSPRING_POPULATION_SIZE,
    GENERATIONS_PER_SWAP,
    MAX_EVALUATIONS,
    AGENTS_NUMBER,
    STARTING_TRUST,
    NO_SEND_PENALTY,
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
    migration_mechanism = "migration" if MIGRATION else "cloning"
    custom_output = f"{OUTPUT_DIR}/{migration_mechanism}_{TRUST_MECHANISM}_starting_trust={STARTING_TRUST}_{start_date}"

    for _ in range(NUMBER_OF_RUNS):
        now = datetime.datetime.now()
        current_date = (
            f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
        )

        for problem in [problem_type(NUM_OF_VARS) for problem_type in PROBLEMS_TO_TEST]:
            """BASE AGENT CLASS SIMULATION"""
            output_file_path = (
                f"{OUTPUT_DIR}/{BaseAgent.name()}_{problem.name()}_{current_date}.csv"
            )
            best_result = run_single_simulation(BaseAgent, problem, output_file_path, None, None)
            print(
                f"Best result for {output_file_path}:",
                best_result,
            )

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

            """ MULTI AGENT CLASS SIMULATION """
            # dir = f"{custom_output}/{problem.name()}"
            # os.makedirs(dir, exist_ok=True)
            # output_file_path = (
            #     f"{dir}/CustomMultiClass_{problem.name()}_{current_date}.csv"
            # )
            # run_single_simulation(
            #     agents, problem, output_file_path, accept_strategies, send_strategies
            # )


def run_single_simulation(
    agent_class,
    problem,
    output_file_path,
    accept_strategy,
    send_strategy,
    crossover_rate=0.5,
    mutation_rate=0.5,
    migration_pop_rate=POPULATION_PART_TO_SWAP,
    migration_interval=GENERATIONS_PER_SWAP,
    save_log=True,
):
    # print(output_file_path)
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
        starting_trust=STARTING_TRUST,
        no_send_penalty=NO_SEND_PENALTY,
        part_to_swap=migration_pop_rate,
        save_log=save_log,
    )
    runner.run_simulation()

    best_result = min(agent.algorithm.result().objectives[0] for agent in runner.agents)
    # print(
    #     f"Best result for {output_file_path}:",
    #     best_result,
    # )
    
    return best_result


def run_irace_compatible_base_simulation(crossover_rate, mutation_rate, migration_pop_rate, migration_interval):
    problem = Griewank(NUM_OF_VARS)  
    return run_single_simulation(BaseAgent, problem, "", None, None, crossover_rate, mutation_rate, migration_pop_rate, migration_interval, save_log=False)


if __name__ == "__main__":
    run_simulations_and_save_results()