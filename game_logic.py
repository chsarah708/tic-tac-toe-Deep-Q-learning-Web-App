import numpy as np
from typing import List, Tuple, Optional


class TicTacToe:
    """
    TicTacToe game class that implements the game logic for training
    a Deep Q-Learning agent to play tic-tac-toe.
    """
    
    def __init__(self):
        """Initialize the game board and game state."""
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for human/agent, -1 for opponent
        self.game_over = False
        self.winner = None
        
    def reset(self):
        """
        Reset the game board and game state to initial values.
        
        Returns:
            np.ndarray: The initial board state (3x3 array of zeros)
        """
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        return self.board.copy()
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state of the board.
        
        Returns:
            np.ndarray: Flattened board state (9-element array)
        """
        return self.board.flatten().copy()
    
    def available_actions(self) -> List[int]:
        """
        Get list of available actions (empty cells) on the board.
        Actions are represented as indices 0-8 (flattened board).
        
        Returns:
            List[int]: List of available action indices
        """
        empty_cells = np.where(self.board.flatten() == 0)[0]
        return empty_cells.tolist()
    
    def make_move(self, action: int, player: int = None) -> bool:
        """
        Make a move on the board at the specified action index.
        
        Args:
            action (int): Action index (0-8) on the flattened board
            player (int, optional): Player making the move (1 or -1). 
                                   Defaults to current_player.
        
        Returns:
            bool: True if move was valid, False if cell was already occupied
        """
        if player is None:
            player = self.current_player
            
        row, col = divmod(action, 3)
        
        if self.board[row, col] != 0:
            return False
        
        self.board[row, col] = player
        self.current_player = -player
        
        # Check for game over
        winner = self.check_winner()
        if winner is not None:
            self.game_over = True
            self.winner = winner
        elif self.is_draw():
            self.game_over = True
            
        return True
    
    def check_winner(self) -> Optional[int]:
        """
        Check if there is a winner on the current board.
        
        Returns:
            Optional[int]: 1 if player 1 won, -1 if player -1 won, None if no winner
        """
        # Check rows
        for row in range(3):
            if self.board[row, 0] == self.board[row, 1] == self.board[row, 2] != 0:
                return self.board[row, 0]
        
        # Check columns
        for col in range(3):
            if self.board[0, col] == self.board[1, col] == self.board[2, col] != 0:
                return self.board[0, col]
        
        # Check diagonals
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return self.board[0, 0]
        
        if self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != 0:
            return self.board[0, 2]
        
        return None
    
    def is_draw(self) -> bool:
        """
        Check if the current board state is a draw (board full with no winner).
        
        Returns:
            bool: True if the board is full and there's no winner, False otherwise
        """
        if len(self.available_actions()) == 0 and self.check_winner() is None:
            return True
        return False
    
    def check_win_flat(self, board_state: np.ndarray) -> Optional[int]:
        """
        Check for a winner on a flattened board state (for evaluation purposes).
        
        Args:
            board_state (np.ndarray): Flattened board state (9-element array)
        
        Returns:
            Optional[int]: 1 if player 1 won, -1 if player -1 won, None if no winner
        """
        board = board_state.reshape(3, 3)
        
        # Check rows
        for row in range(3):
            if board[row, 0] == board[row, 1] == board[row, 2] != 0:
                return board[row, 0]
        
        # Check columns
        for col in range(3):
            if board[0, col] == board[1, col] == board[2, col] != 0:
                return board[0, col]
        
        # Check diagonals
        if board[0, 0] == board[1, 1] == board[2, 2] != 0:
            return board[0, 0]
        
        if board[0, 2] == board[1, 1] == board[2, 0] != 0:
            return board[0, 2]
        
        return None
    
    def smart_logic(self, player: int = -1) -> int:
        """
        Implement smart AI logic for the opponent.
        Uses heuristics to:
        1. Win if possible
        2. Block opponent from winning
        3. Take center if available
        4. Take corners
        5. Take edges
        
        Args:
            player (int): The player to compute the move for (1 or -1)
        
        Returns:
            int: The best action index (0-8)
        """
        available = self.available_actions()
        
        if not available:
            return None
        
        # Try to win
        for action in available:
            board_copy = self.board.copy()
            row, col = divmod(action, 3)
            board_copy[row, col] = player
            
            # Check if this move wins
            temp_board = self.board.copy()
            self.board = board_copy
            if self.check_winner() == player:
                self.board = temp_board
                return action
            self.board = temp_board
        
        # Try to block opponent
        opponent = -player
        for action in available:
            board_copy = self.board.copy()
            row, col = divmod(action, 3)
            board_copy[row, col] = opponent
            
            temp_board = self.board.copy()
            self.board = board_copy
            if self.check_winner() == opponent:
                self.board = temp_board
                return action
            self.board = temp_board
        
        # Take center if available
        if 4 in available:
            return 4
        
        # Take corners (0, 2, 6, 8)
        corners = [0, 2, 6, 8]
        corner_available = [c for c in corners if c in available]
        if corner_available:
            return corner_available[0]
        
        # Take edges (1, 3, 5, 7)
        return available[0]
    
    def render_board(self) -> str:
        """
        Render the current board state as a formatted string for display.
        
        Returns:
            str: Formatted board representation
        """
        symbol_map = {
            0: ' ',
            1: 'X',
            -1: 'O'
        }
        
        board_str = "\n"
        board_str += "     0   1   2\n"
        board_str += "   +---+---+---+\n"
        
        for row in range(3):
            board_str += f" {row} | "
            for col in range(3):
                board_str += symbol_map[self.board[row, col]] + " | "
            board_str += "\n"
            board_str += "   +---+---+---+\n"
        
        return board_str
    
    def get_game_status(self) -> str:
        """
        Get the current game status as a string.
        
        Returns:
            str: Game status message
        """
        if self.game_over:
            if self.winner == 1:
                return "Player 1 (X) wins!"
            elif self.winner == -1:
                return "Player 2 (O) wins!"
            else:
                return "It's a draw!"
        else:
            player_symbol = "X" if self.current_player == 1 else "O"
            return f"Current player: {player_symbol}"
    
    def copy(self):
        """
        Create a deep copy of the current game state.
        
        Returns:
            TicTacToe: A new TicTacToe instance with copied state
        """
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        return new_game
