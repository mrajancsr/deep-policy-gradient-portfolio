@dataclass
class EnsembleDDPGAgent:
    portfolio: Portfolio
    n_agents: int
    batch_size: int
    window_size: int
    step_size: int
    n_iter: int
    learning_rate: Optional[float] = 3e-5
    betas: Optional[Tuple[float, float]] = (0.0, 0.9)
    device: Optional[str] = "mps"
    agents: list = field(init=False)
    pvm_ensemble: list = field(init=False)

    def __post_init__(self):
        self.agents = [
            DDPGAgent(
                portfolio=self.portfolio,
                batch_size=self.batch_size,
                window_size=self.window_size,
                step_size=self.step_size,
                n_iter=self.n_iter,
                learning_rate=self.learning_rate,
                betas=self.betas,
                device=self.device,
            )
            for _ in range(self.n_agents)
        ]
        self.pvm_ensemble = [agent.pvm for agent in self.agents]

    def pre_train_ensemble(self):
        """Pretrain each agent and fill their replay buffers."""
        for idx, agent in enumerate(self.agents):
            print(f"Pre-training agent {idx + 1}/{self.n_agents}...")
            agent.pre_train()

    def train_ensemble(self, n_episodes: int = 50, n_iterations_per_episode: int = 20):
        """Train each agent independently."""
        for idx, agent in enumerate(self.agents):
            print(f"Training agent {idx + 1}/{self.n_agents}...")
            agent.train(n_episodes=n_episodes, n_iterations_per_episode=n_iterations_per_episode)

    def get_ensemble_action(self, state: Tuple[torch.tensor, torch.tensor]):
        """Combine actions from all agents using an ensemble method."""
        actions = torch.stack(
            [agent.select_action(state, exploration=False) for agent in self.agents]
        )
        return torch.mean(actions, dim=0)  # Simple averaging

    def evaluate_ensemble(self, dataset: DataLoader):
        """Evaluate the ensemble performance."""
        portfolio_values = []
        for i, (xt, prev_index) in enumerate(dataset):
            previous_action = self.pvm_ensemble[0].get_memory_stack(prev_index)  # Use one agent's PVM
            state = (xt, previous_action)
            ensemble_action = self.get_ensemble_action(state)
            portfolio_values.append(self.portfolio.get_portfolio_value(ensemble_action))
        return portfolio_values
    
    def combine_replay_buffers(self):
    combined_experiences = []
    for agent in self.agents:
        combined_experiences.extend(agent.replay_memory.sample(len(agent.replay_memory)))
    
    for agent in self.agents:
        agent.replay_memory.add_experiences(combined_experiences)

# use the ensemble
# Initialize the ensemble
ensemble_agent = EnsembleDDPGAgent(
    portfolio=portfolio,
    n_agents=3,  # Number of agents
    batch_size=50,
    window_size=50,
    step_size=1,
    n_iter=100,
    device="mps",
)

# Pre-train each agent
ensemble_agent.pre_train_ensemble()

# Train each agent
ensemble_agent.train_ensemble(n_episodes=50, n_iterations_per_episode=20)

# Evaluate ensemble
portfolio_values = ensemble_agent.evaluate_ensemble(kraken_ds)
