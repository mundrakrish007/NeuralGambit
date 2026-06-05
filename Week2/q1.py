import json
import copy  # use it for deepcopy if needed
import math  # for math.inf
import logging

logging.basicConfig(format='%(levelname)s - %(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

# Global variables in which you need to store player strategies (this is data structure that'll be used for evaluation)
# Mapping from histories (str) to probability distribution over actions
strategy_dict_x = {}
strategy_dict_o = {}

class History:
    def __init__(self, history=None):
        """
        # self.history : Eg: [0, 4, 2, 5] keeps track of sequence of actions played since the beginning of the game.
        Each action is an integer between 0-8 representing the square in which the move will be played as shown below.
          ___ ___ ____
         |_0_|_1_|_2_|
         |_3_|_4_|_5_|
         |_6_|_7_|_8_|

        # self.board empty squares are represented using '0' and occupied squares are either 'x' or 'o'.
        Eg: ['x', '0', 'x', '0', 'o', 'o', '0', '0', '0'] for board
          ___ ___ ____
         |_x_|___|_x_|
         |___|_o_|_o_|
         |___|___|___|

        # self.player: 'x' or 'o'
        Player whose turn it is at the current history/board

        :param history: list keeps track of sequence of actions played since the beginning of the game.
        """
        if history is not None:
            self.history = history
            self.board = self.get_board()
        else:
            self.history = []
            self.board = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
        self.player = self.current_player()

    def current_player(self):
        """ Player function
        Get player whose turn it is at the current history/board
        :return: 'x' or 'o' or None
        """
        total_num_moves = len(self.history)
        if total_num_moves < 9:
            if total_num_moves % 2 == 0:
                return 'x'
            else:
                return 'o'
        else:
            return None

    def get_board(self):
        """
        Play out the current self.history and get the board corresponding to the history in self.board.
        :return: list Eg: ['x', '0', 'x', '0', 'o', 'o', '0', '0', '0']
        """
        board = ['0', '0', '0', '0', '0', '0', '0', '0', '0']
        for i in range(len(self.history)):
            if i % 2 == 0:
                board[self.history[i]] = 'x'
            else:
                board[self.history[i]] = 'o'
        return board

    def is_win(self):
        """
        Check if the board position is a win for either player.
        :return: 'x' if x wins, 'o' if o wins, else None
        """
        vertical_win = (self.board[0] == self.board[3] == self.board[6] != '0') or \
                       (self.board[1] == self.board[4] == self.board[7] != '0') or \
                       (self.board[2] == self.board[5] == self.board[8] != '0')
        horizontal_win = (self.board[0] == self.board[1] == self.board[2] != '0') or \
                         (self.board[3] == self.board[4] == self.board[5] != '0') or \
                         (self.board[6] == self.board[7] == self.board[8] != '0')
        diagonal_win = (self.board[0] == self.board[4] == self.board[8] != '0') or \
                       (self.board[2] == self.board[4] == self.board[6] != '0')
        return vertical_win or horizontal_win or diagonal_win
                
        return None

    def is_draw(self):
        """
        Check if the board position is a draw.
        """
        # It's a draw if there is no winner and the board is full (no '0' left)
        if self.is_win() is None and '0' not in self.board:
            return True
        return False

    def get_valid_actions(self):
        """
        Get the empty squares from the board.
        :return: List of valid actions (integers 0-8)
        """
        valid_actions = []
        for i in range(9):
            if self.board[i] == '0':
                valid_actions.append(i)
        return valid_actions

    def is_terminal_history(self):
        """
        Check if the history is a terminal history (the game is over).
        """
        # Game is over if someone won or it is a draw
        if self.is_win() is not None or self.is_draw():
            return True
        return False

    def get_utility_given_terminal_history(self):
        """
        Get the utility score for a terminal board state.
        Assuming 'x' wants to maximize (+1) and 'o' wants to minimize (-1).
        Draw is 0.
        """
        winner = self.is_win()
        if winner == 'x':
            return 1.0
        elif winner == 'o':
            return -1.0
        else:
            return 0.0

    def update_history(self, action):
        """
        Create a deepcopy and update the history obj to get the next history object.
        """
        # Create a new history list by copying the current one and adding the new action
        new_history_list = copy.deepcopy(self.history)
        new_history_list.append(action)
        # Return a new History object
        return History(new_history_list)

def backward_induction(history_obj):
    """
    :param history_obj: Histroy class object
    :return: best achievable utility (float) for th current history_obj
    
    """
    global strategy_dict_x, strategy_dict_o

    # 1. Base case: If the game is over, return the utility of this state
    if history_obj.is_terminal_history():
        return history_obj.get_utility_given_terminal_history()

    # 2. Get current player and valid actions
    player = history_obj.current_player()
    valid_actions = history_obj.get_valid_actions()

    # 3. Setup variables to find the best move
    # 'x' wants to maximize the score, so start with the lowest possible number
    # 'o' wants to minimize the score, so start with the highest possible number
    if player == 'x':
        best_utility = -math.inf
    else:
        best_utility = math.inf
        
    best_action = None

    # Initialize a dictionary for this state's policy: keys "0" to "8", default value 0.0
    policy = {str(i): 0.0 for i in range(9)}

    # 4. Try all valid actions to see which leads to the best outcome
    for action in valid_actions:
        next_history_obj = history_obj.update_history(action)
        
        # Recursively find the utility of the child state
        utility = backward_induction(next_history_obj)

        if player == 'x':
            # Maximize for 'x'
            if utility > best_utility:
                best_utility = utility
                best_action = action
        else:
            # Minimize for 'o'
            if utility < best_utility:
                best_utility = utility
                best_action = action

    # 5. Set the probability of the best action to 1 (Deterministic Strategy)
    if best_action is not None:
        policy[str(best_action)] = 1.0

    # 6. Save the policy into the global dictionaries
    # Convert history list (e.g. [0, 4, 2]) to string "042" to use as a dictionary key
    history_str = "".join(str(move) for move in history_obj.history)
    
    if player == 'x':
        strategy_dict_x[history_str] = policy
    else:
        strategy_dict_o[history_str] = policy

    return best_utility

def solve_tictactoe():
    # Start backward induction from a blank board
    backward_induction(History())
    
    # Save results to json files
    with open('./policy_x.json', 'w') as f:
        json.dump(strategy_dict_x, f)
    with open('./policy_o.json', 'w') as f:
        json.dump(strategy_dict_o, f)
    
    return strategy_dict_x, strategy_dict_o

if __name__ == "__main__":
    logging.info("Start")
    solve_tictactoe()
    logging.info("End")
