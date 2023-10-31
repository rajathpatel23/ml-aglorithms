import numpy as np

# Define the payoff matrix
payoff_matrix = np.array([[2, 0], [3, 1]])

# Define the strategies for each player
player1_strategies = ['A', 'B']
player2_strategies = ['C', 'D']

# Define a function to calculate the payoffs for each player given their strategies
def calculate_payoffs(player1_strategy, player2_strategy):
    row_index = player1_strategies.index(player1_strategy)
    col_index = player2_strategies.index(player2_strategy)
    return (payoff_matrix[row_index][col_index], payoff_matrix[col_index][row_index])

# Define a function to simulate a game between two players with given strategies
def play_game(player1_strategy, player2_strategy):
    player1_payoff, player2_payoff = calculate_payoffs(player1_strategy, player2_strategy)
    return (player1_payoff, player2_payoff)

# Define a function to calculate the average payoff for a given strategy by playing it against all other strategies
def calculate_average_payoff(strategy, opponent_strategies):
    total_payoff = 0
    for opponent_strategy in opponent_strategies:
        player1_payoff, player2_payoff = play_game(strategy, opponent_strategy)
        total_payoff += player1_payoff
    return total_payoff / len(opponent_strategies)

# Define a function to find the best response for a given player given their opponent's strategy
def find_best_response(player_strategies, opponent_strategy):
    best_strategy = None
    best_payoff = float('-inf')
    for strategy in player_strategies:
        player1_payoff, player2_payoff = play_game(strategy, opponent_strategy)
        if player1_payoff > best_payoff:
            best_strategy = strategy
            best_payoff = player1_payoff
    return best_strategy

# Define the strategies to analyze
strategies_to_analyze = ['A', 'B', 'C', 'D']

# Calculate the average payoff for each strategy when played against all other strategies
average_payoffs = []
for strategy in strategies_to_analyze:
    opponent_strategies = [s for s in strategies_to_analyze if s != strategy]
    average_payoff = calculate_average_payoff(strategy, opponent_strategies)
    average_payoffs.append(average_payoff)

# Print the results
for i, strategy in enumerate(strategies_to_analyze):
    print(f"Strategy {strategy} has an average payoff of {average_payoffs[i]:.2f}")

# Find the best response for each strategy given an opponent's strategy
opponent_strategy = 'C'
for strategy in strategies_to_analyze:
    best_response = find_best_response(player1_strategies, opponent_strategy)
    print(f"The best response for Player 1 when the opponent plays {opponent_strategy} is {best_response}")
