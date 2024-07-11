from src.models.agent import BaseAgent


class Agent(BaseAgent):

    def train(self, start_idx: int, end_idx: int, training_epsidoes: int, epsilon_decya_func, initial_epsilon,
              final_epsilon):
        pass

    def test(self, start_idx: int, end_idx: int):
        pass

    def _act(self, step_type):
        pass




