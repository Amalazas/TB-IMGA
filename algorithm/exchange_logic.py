import random
from numpy import choice as np_choice
import numpy as np
import pandas as pd
from typing import Type, Sequence

from .agents.base import BaseAgent
from collections import defaultdict

from analysis.constants_and_params import (
    MIGRATION_POLICY, 
    MigrationPolicy, 
    AUCTION_TRUST_WEIGHT, 
    AUCTION_SOLUTION_WEIGHT,
    )

from algorithm.agents.strategy_based import AcceptStrategy


class ExchangeMarket:
    def __init__(
        self,
        agents: Sequence[Type[BaseAgent]],
        migration: bool = False,
    ):
        self.migration = migration
        self.log = {}
        self.agents = agents
        self.id2agent = defaultdict(list)
        for agent in self.agents:
            self.id2agent[agent.id].append(agent)


    def exchange_information(self):
        
        ### Agent pairing
        paired_agents = []
        pair_string = ""
        if 'pairs' not in self.log:
                self.log['pairs'] = []

        # Pairing based on random selection
        if MIGRATION_POLICY is MigrationPolicy.Basic:
            shuffled_agent_list_ids = list(range(len(self.agents)))
            random.shuffle(shuffled_agent_list_ids)
            for i in range(0, len(shuffled_agent_list_ids), 2):
                paired_agents.append( 
                    (self.agents[shuffled_agent_list_ids[i]], self.agents[shuffled_agent_list_ids[i + 1]])
                )
                # Logging
                pair_string += f"{shuffled_agent_list_ids[i]}:{shuffled_agent_list_ids[i + 1]}_"
            self.log['pairs'].append(pair_string[:-1])
                    
        # Pairing based on random trust-weighted selection
        elif MIGRATION_POLICY is MigrationPolicy.TrustBasedRoulette:
            roulette_string = ""
            if 'roulette' not in self.log:
                self.log['roulette'] = []
            agent_ids = [ agent.id for agent in self.agents ]
            while len(agent_ids) > 1:
                # Select a base agent randomly
                base_agent_id = random.choice(agent_ids)
                agent_ids.remove(base_agent_id)
                # Calculate trust weights
                trust_weights = []
                trust_agent_ids = []
                trust_sum = 0
                for agent_id, trust in self.id2agent[base_agent_id].trust.items():
                    if agent_id in agent_ids:
                        trust_weights.append(trust)
                        trust_agent_ids.append(agent_id)
                        trust_sum += trust
                trust_weights = [weight / trust_sum for weight in trust_weights]
                # Roulette wheel selection
                paired_agent_id = np_choice(trust_agent_ids, 1, p=trust_weights)[0]
                # Remove the paired agent from the list of available agents and save the pair
                agent_ids.remove(paired_agent_id)
                paired_agents.append(
                    (self.id2agent[base_agent_id], self.id2agent[paired_agent_id])
                    )
                # Logging
                pair_string += f"{base_agent_id}:{paired_agent_id}_"    
                roulette_string += f"{base_agent_id}:"
                for id, weight in sorted( [(trust_agent_ids[i], trust_weights[i]) for i in range(len(trust_weights))] , reverse=True, key=lambda x: x[1]):
                    roulette_string += f"<{id}-{weight:.2f}>_"
                roulette_string = roulette_string[:-1] + '|'
            self.log['pairs'].append(pair_string[:-1])
            self.log['roulette'].append(roulette_string[:-1])

        # Pairing based on trust and quality auction
        elif MIGRATION_POLICY is MigrationPolicy.TrustBasedAuction:
            auction_string = ""
            if 'auction' not in self.log:
                self.log['auction'] = []
            agent_ids = [ agent.id for agent in self.agents ]
            while len(agent_ids) > 1:
                ### Select a base agent randomly
                base_agent_id = random.choice(agent_ids)
                agent_ids.remove(base_agent_id)
                ### Collecting proposals from other agents and their trust towards the selected base agent
                proposed_solutions = [ 
                    (agent.id, agent.get_solutions_to_share(self.id2agent[base_agent_id])) for agent in self.agents if agent.id in agent_ids 
                    ]
                trust_towards_base_agent = [
                    (agent.id, agent.trust[base_agent_id]) for agent in self.agents if agent.id in agent_ids
                ]
                ### Normalization of trust and quality scores plus auction value calculation (min-max normalization)
                # Trust
                trust_min, trust_max = min(trust_towards_base_agent, key=lambda x: x[1])[1], max(trust_towards_base_agent, key=lambda x: x[1])[1]
                normalized_trust = [
                    (agent_id, (trust - trust_min) / (trust_max - trust_min)) for agent_id, trust in trust_towards_base_agent
                ]
                # Quality
                normalized_quality = []
                if self.id2agent[base_agent_id].accept_strategy is AcceptStrategy.Better:
                    # Fitness Scores - the lower the better
                    avg_solution_scores = []        
                    for agent_id, solutions in proposed_solutions:
                        avg_score = sum(solution.objectives[0] for solution in solutions) / len(solutions) if solutions else 0
                        avg_solution_scores.append((agent_id, avg_score))
                    score_min, score_max = min(avg_solution_scores, key=lambda x: x[1])[1], max(avg_solution_scores, key=lambda x: x[1])[1]
                    # normalized_quality equals (1 - normalized_score) to reverse the scale, as lower scores are better
                    normalized_quality = [
                        (agent_id, 1-((score - score_min) / (score_max - score_min)) ) for agent_id, score in avg_solution_scores
                    ]
                elif self.id2agent[base_agent_id].accept_strategy is AcceptStrategy.Different:
                    # Diversity Scores - the absolute value of the dot product calculated on the mean base agent vector and mean proposed solution vector - the higher the value, the better
                    avg_solution_diversity = []
                    base_agent_variables_mean = np.array([solution.variables for solution in self.id2agent[base_agent_id].solutions]).mean(axis=0)
                    for agent_id, solutions in proposed_solutions:
                        dot_products = [ np.dot(solution.variables, base_agent_variables_mean) for solution in solutions ]
                        avg_diversity = np.sum(np.abs(dot_products)) / len(solutions) if solutions else 0
                        avg_solution_diversity.append((agent_id, avg_diversity))
                    diversity_min, diversity_max = min(avg_solution_diversity, key=lambda x: x[1])[1], max(avg_solution_diversity, key=lambda x: x[1])[1]
                    normalized_quality = [
                        (agent_id, (diversity - diversity_min) / (diversity_max - diversity_min)) for agent_id, diversity in avg_solution_diversity
                    ]
                ### Selection of the best agent based on auction value and pairing it with the base agent + save the pair
                agent_bids = [ (agent_id, AUCTION_TRUST_WEIGHT * trust + AUCTION_SOLUTION_WEIGHT * quality)
                                for (agent_id, trust), (_, quality) in zip(normalized_trust, normalized_quality) ]
                best_agent_id = max(agent_bids, key=lambda x: x[1])[0]
                paired_agents.append((self.id2agent[base_agent_id], self.id2agent[best_agent_id]))
                agent_ids.remove(best_agent_id)
                # Logging
                auction_string += f"{base_agent_id}:"
                for id, bid in sorted(trust_weights, reverse=True, key=lambda x: x[1]):
                    auction_string += f"({id}-{bid:.2f})_"
                auction_string = auction_string[:-1] + '|'
                pair_string += f"{base_agent_id}:{best_agent_id}_"  
            self.log['pairs'].append(pair_string[:-1])
            self.log['auction'].append(auction_string[:-1])

        ### Migration
        for agent1, agent2 in paired_agents:
            starting_population_size = len(agent1.algorithm.solutions)
            agent1_solutions = agent1.get_solutions_to_share(agent2)
            agent2_solutions = agent2.get_solutions_to_share(agent1)
            if self.migration:
                agent1.remove_solutions(agent1_solutions)
                agent2.remove_solutions(agent2_solutions)
            agent1.use_shared_solutions(agent2_solutions, agent2, starting_population_size)
            agent2.use_shared_solutions(agent1_solutions, agent1, starting_population_size)


    def save_log(self, log_file_path: str):
        pd.DataFrame(self.log).to_csv(log_file_path, index=False)