def get_reward_with_drawdown(wt, yt, wt_prev, portfolio_value, max_portfolio_value):
    """
    Reward function with drawdown penalty.
    """
    # Calculate the portfolio return
    portfolio_return = yt.dot(wt_prev)

    # Update the maximum portfolio value
    max_portfolio_value = max(max_portfolio_value, portfolio_value)

    # Calculate drawdown as a percentage
    drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value

    # Reward shaping with drawdown penalty
    reward = torch.log(portfolio_return + 1e-8)
    shaped_reward = reward - 0.1 * drawdown  # Adjust penalty weight as needed

    return shaped_reward, max_portfolio_value


def train_critic_with_drawdown(self, experience, is_weights, max_portfolio_value):
    """
    Train the critic while incorporating a drawdown penalty.
    """
    self.critic_optimizer.zero_grad()

    # Extract states, actions, and rewards
    xt, previous_noncash_actions = experience.state
    reward = experience.reward
    noncash_actions = experience.action

    # Update portfolio value and calculate drawdown
    portfolio_value = self.portfolio.calculate_value(noncash_actions)
    max_portfolio_value = max(max_portfolio_value, portfolio_value)
    drawdown = (max_portfolio_value - portfolio_value) / max_portfolio_value

    # Compute Q-values
    state = (xt, previous_noncash_actions)
    predicted_q_values = self.critic(state, noncash_actions)

    with torch.no_grad():
        next_q_values = self.target_critic(experience.next_state, noncash_actions)
        td_target = reward + self.gamma * next_q_values

    # Add drawdown penalty to critic loss
    td_error = td_target - predicted_q_values
    critic_loss = (td_error**2).mean() + 0.1 * drawdown  # Adjust penalty weight

    critic_loss.backward()
    self.critic_optimizer.step()

    return td_error, critic_loss.item(), max_portfolio_value


def train_actor_with_smoothness(self, experience, is_weights):
    self.actor_optimizer.zero_grad()

    logits = self.actor(experience.state)
    predicted_actions = torch.softmax(logits, dim=1)

    # Compute Q-values
    q_values = self.critic(experience.state, predicted_actions)
    actor_loss = -q_values.mean()

    # Add smoothness penalty
    wt_prev = experience.state[1]  # Previous weights
    smoothness_penalty = torch.sum(torch.abs(predicted_actions - wt_prev), dim=1).mean()
    actor_loss += 0.01 * smoothness_penalty  # Adjust penalty weight

    actor_loss.backward()
    self.actor_optimizer.step()

    return actor_loss.item()


# During evaluation
portfolio_values = calculate_portfolio_values(actions, returns)
max_drawdown = calculate_max_drawdown(portfolio_values)

# Evaluate performance
total_return = portfolio_values[-1] - portfolio_values[0]
evaluation_score = total_return - 0.1 * max_drawdown  # Adjust weight
