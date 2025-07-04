import random
from numpy import choice as np_choice
import numpy as np
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
        self.agents = agents
        self.id2agent = defaultdict(list)
        for agent in self.agents:
            self.id2agent[agent.id].append(agent)

    def exchange_information(self):
        
        ### Agent pairing
        paired_agents = []

        # Pairing based on random selection
        if MIGRATION_POLICY is MigrationPolicy.Basic:
            shuffled_agent_list_ids = list(range(len(self.agents)))
            random.shuffle(shuffled_agent_list_ids)
            for i in range(0, len(shuffled_agent_list_ids), 2):
                paired_agents.append( 
                    (self.agents[shuffled_agent_list_ids[i]], self.agents[shuffled_agent_list_ids[i + 1]])
                )
        
        # Pairing based on random trust-weighted selection
        elif MIGRATION_POLICY is MigrationPolicy.TrustBasedRoulette:
            agent_ids = [ agent.id for agent in self.agents ]
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
            paired_agents.append(self.id2agent[base_agent_id], self.id2agent[paired_agent_id])

        # Pairing based on trust and quality auction
        elif MIGRATION_POLICY is MigrationPolicy.TrustBasedAuction:

            ### Losowy wybór agenta bazowego
            agent_ids = [ agent.id for agent in self.agents ]
            base_agent_id = random.choice(agent_ids)
            agent_ids.remove(base_agent_id)

            ### Zebranie propozycji rozwiązań od innych agentów oraz ich zaufania do wylosowanego agenta bazowego
            proposed_solutions = [ 
                (agent.id, agent.get_solutions_to_share(self.id2agent[base_agent_id])) for agent in self.agents if agent.id in agent_ids 
                ]
            trust_towards_base_agent = [
                (agent.id, agent.trust[base_agent_id]) for agent in self.agents if agent.id in agent_ids
            ]

            ### Normalizacja zaufania, jakości rozwiązań i wyliczenia wartości licytacji (min-max normalization)
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
                # Diversity Scores - wartość bezwzględna odchylenia od średniej - im większa wartość, tym lepiej
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
                
            ### Wybranie najlepszego agenta na podstawie wartości licytacji i sparowanie go z agentem bazowym + save the pair
            agent_bids = [ (agent_id, AUCTION_TRUST_WEIGHT * trust + AUCTION_SOLUTION_WEIGHT * quality)
                            for (agent_id, trust), (_, quality) in zip(normalized_trust, normalized_quality) ]
            best_agent_id = max(agent_bids, key=lambda x: x[1])[0]
            paired_agents.append((self.id2agent[base_agent_id], self.id2agent[best_agent_id]))
            agent_ids.remove(best_agent_id)

        ### TODO: Migration
        # starting_population_size = len(agent.algorithm.solutions)
        # paired_agent_shared_solutions = paired_agent.get_solutions_to_share(agent)
        # agent_shared_solutions = agent.get_solutions_to_share(paired_agent)
        # if self.migration:
        #     paired_agent.remove_solutions(paired_agent_shared_solutions)
        #     agent.remove_solutions(agent_shared_solutions)
        # agent.use_shared_solutions(
        #     paired_agent_shared_solutions, paired_agent, starting_population_size
        # )
        # paired_agent.use_shared_solutions(
        #     agent_shared_solutions, agent, starting_population_size
        # )
