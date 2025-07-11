import time

from typing import Callable, Type, Optional, List, Tuple

import pandas as pd

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.agents.strategy_based import TrustMechanism
from analysis.constants_and_params import POPULATION_SIZE, RESTART_TRUST_THRESHOLD, RESTARTING_ENABLED

from .agents import AcceptStrategy, BaseAgent, SendStrategy, StrategyAgent
from .exchange_logic import ExchangeMarket


class Runner:
    def __init__(
        self,
        output_file_path: str,
        agent_class: (
            Callable[[GeneticAlgorithm], Type[BaseAgent]]
            | List[Callable[[GeneticAlgorithm], Type[BaseAgent]]]
        ),
        agents_number: Optional[int],
        generations_per_swap: int,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        solution_comparator: Comparator = ObjectiveComparator(0),
        accept_strategy: Optional[AcceptStrategy | List[AcceptStrategy]] = None,
        send_strategy: Optional[SendStrategy | List[AcceptStrategy]] = None,
        trust_mechanism: Optional[TrustMechanism] = None,
        starting_trust: Optional[int] = None,
        no_send_penalty: Optional[int] = 2,
        part_to_swap: Optional[float] = 0.1,
        migration: bool = True,
    ):
        global_trust = {agent_id: starting_trust for agent_id in range(agents_number)} # Initial trust values for agents
        # In case of a Uniform Agent Class simulation
        if callable(agent_class):
            self.agents = [
                agent_class(
                    GeneticAlgorithm(
                        problem,
                        population_size,
                        offspring_population_size,
                        mutation,
                        crossover,
                        selection,
                        termination_criterion,
                        population_generator,
                        population_evaluator,
                        solution_comparator,
                    ),
                    send_strategy,
                    accept_strategy,
                    trust_mechanism,
                    global_trust,
                    starting_trust=starting_trust,
                    id=agent_nr,
                )
                for agent_nr in range(agents_number)
            ]
        # In case of a Multiple Agent Class simulation (lists of Agent Classes and Send/Accept Strategies were passed as args)
        elif isinstance(agent_class, list):
            self.agents = [
                agent_class[agent_nr](
                    GeneticAlgorithm(
                        problem,
                        population_size,
                        offspring_population_size,
                        mutation,
                        crossover,
                        selection,
                        termination_criterion,
                        population_generator,
                        population_evaluator,
                        solution_comparator,
                    ),
                    send_strategy[agent_nr],
                    accept_strategy[agent_nr],
                    trust_mechanism,
                    global_trust,
                    starting_trust=starting_trust,
                    no_send_penalty=no_send_penalty,
                    part_to_swap=part_to_swap,
                    id=agent_nr,
                )
                for agent_nr in range(len(agent_class))
            ]

        self.exchange_market = ExchangeMarket(self.agents, migration)
        self.generations_per_swap = generations_per_swap
        self.output_file_path = output_file_path


    def restart_criterion_met(self) -> Tuple[bool, int]:
        ### OLD IDEA, CURRENTLY NOT USED
        ### Go through all agent and check whether there is an agent whose average towards it is above the threshold
        ### (higher value in the agent.trust dictionary means lower trust level towards the agent)
        
        average_trusts_per_agent = { agent.id: 0.0 for agent in self.agents }
        for agent in self.agents:
            for agent_id, trust in agent.trust.items():
                average_trusts_per_agent[agent_id] += trust        
        average_trusts_per_agent = {
            agent_id: trust / len(self.agents) for agent_id, trust in average_trusts_per_agent.items()
        }
        
        ### Get maximum value and it's key from the average trust values 
        worst_trust_agent_id = max(average_trusts_per_agent, key=average_trusts_per_agent.get)
        worst_trust_value = average_trusts_per_agent[worst_trust_agent_id]
        
        if worst_trust_value > RESTART_TRUST_THRESHOLD:
            return True, worst_trust_agent_id
        else:
            return False, -1


    def restart_agent(self, agent_id) -> None:
        ### OLD IDEA, CURRENTLY NOT USED
        pass

    
    def run_simulation(self):
        start_computing_time = time.time()

        for agent in self.agents:
            agent.algorithm.solutions = agent.algorithm.create_initial_solutions()

        for agent in self.agents:
            agent.algorithm.solutions = agent.algorithm.evaluate(
                agent.algorithm.solutions
            )

        for agent in self.agents:
            agent.algorithm.init_progress()

        data_to_save = {
            "generation": [],
            "agent_id": [],
            "score": [],
            "class": [],
            "trust": [],
        }

        # TODO: update this to make sense with more compilcated criteria than number of evaluations.
        number_of_generations = 0
        while not agent.algorithm.stopping_condition_is_met():
            number_of_generations += 1
            for agent_id, agent in enumerate(self.agents):
                try:
                    agent.algorithm.step()
                    agent.algorithm.update_progress()
                    data_to_save["generation"].append(number_of_generations)
                    data_to_save["agent_id"].append(agent_id)
                    data_to_save["score"].append(agent.algorithm.result().objectives[0])
                    assert len(agent.algorithm.solutions) == POPULATION_SIZE
                    if isinstance(agent, StrategyAgent):
                        data_to_save["class"].append(
                            agent.accept_strategy.name + "_" + agent.send_strategy.name
                        )
                        trust_string = ""
                        for trust_agent_id, trust_level in agent.trust.items():
                            trust_string += f"{trust_agent_id}:{int(trust_level)}_"
                        data_to_save["trust"].append(trust_string[:-1])
                    else:
                        data_to_save["class"].append(type(agent).__name__)
                        data_to_save["trust"].append("not_applicable")
                except KeyboardInterrupt:
                    pd.DataFrame(data_to_save).to_csv(
                        self.output_file_path, index=False
                    )
                    print("Program stopped by user.")
                    exit()
                except Exception as e:
                    pd.DataFrame(data_to_save).to_csv(
                        self.output_file_path, index=False
                    )
                    self.exchange_market.save_log("." + ''.join(self.output_file_path.split('.')[:-1]) + "_exchange_log.csv")
                    print(f"An error occurred: {e}")
                    print("Program stopped due to an error.")
                    exit()

            if number_of_generations % self.generations_per_swap == 0:
                self.exchange_market.exchange_information()
                # if RESTARTING_ENABLED:
                #     criterion_met, agent_id = self.restart_criterion_met()
                #     if criterion_met:
                #         self.restart_agent(agent_id)

        total_computing_time = time.time() - start_computing_time

        pd.DataFrame(data_to_save).to_csv(self.output_file_path, index=False)
        self.exchange_market.save_log("." + ''.join(self.output_file_path.split('.')[:-1]) + "_exchange_log.csv")

        for agent in self.agents:
            agent.algorithm.start_computing_time = start_computing_time
            agent.algorithm.total_computing_time = total_computing_time
