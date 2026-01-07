import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path


class TicTacToeGame:
    """Tic Tac Toe game implementation for human vs AI play."""
    
    def __init__(self):
        """Initialize the game board and game state."""
        self.board = np.zeros((3, 3), dtype=int)  # 0: empty, 1: human, -1: AI
        self.game_over = False
        self.winner = None
    
    def reset(self):
        """Reset the board for a new game."""
        self.board = np.zeros((3, 3), dtype=int)
        self.game_over = False
        self.winner = None
    
    def is_valid_move(self, row, col):
        """Check if a move is valid."""
        if row < 0 or row > 2 or col < 0 or col > 2:
            return False
        return self.board[row, col] == 0
    
    def make_move(self, row, col, player):
        """Place a piece on the board."""
        if self.is_valid_move(row, col):
            self.board[row, col] = player
            return True
        return False
    
    def check_winner(self):
        """Check if there's a winner or draw."""
        # Check rows
        for row in self.board:
            if abs(row.sum()) == 3:
                return row[0]
        
        # Check columns
        for col in self.board.T:
            if abs(col.sum()) == 3:
                return col[0]
        
        # Check diagonals
        if abs(self.board.diagonal().sum()) == 3:
            return self.board[0, 0]
        if abs(np.fliplr(self.board).diagonal().sum()) == 3:
            return self.board[0, 2]
        
        # Check for draw
        if not np.any(self.board == 0):
            return 0  # Draw
        
        return None  # Game still in progress
    
    def get_available_moves(self):
        """Get list of available moves."""
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i, j] == 0:
                    moves.append((i, j))
        return moves
    
    def board_to_state(self):
        """Convert board to a flattened state for the neural network."""
        return self.board.flatten().astype(np.float32)
    
    def display_board(self):
        """Display the current board state."""
        print("\nCurrent Board:")
        for i, row in enumerate(self.board):
            display_row = []
            for j, cell in enumerate(row):
                if cell == 1:
                    display_row.append("X")  # Human
                elif cell == -1:
                    display_row.append("O")  # AI
                else:
                    display_row.append(str(i * 3 + j + 1))  # Position number
            print(f" {display_row[0]} | {display_row[1]} | {display_row[2]} ")
            if i < 2:
                print("-----------")
        print()


class DQNPlayer:
    """AI player using Deep Q-Network."""
    
    def __init__(self, model_path=None):
        """
        Initialize the DQN player and load the model.
        
        Args:
            model_path: Path to the saved DQN model. If None, uses default path.
        """
        self.model = None
        self.model_path = model_path or self._get_default_model_path()
        self.load_model()
    
    def _get_default_model_path(self):
        """Get the default model path."""
        current_dir = Path(__file__).parent
        return os.path.join(current_dir, "models", "dqn_model.h5")
    
    def load_model(self):
        """Load the DQN model from disk."""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                print(f"✓ DQN model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"✗ Error loading model from {self.model_path}: {e}")
                print("Using random moves instead...")
                self.model = None
        else:
            print(f"✗ Model file not found at {self.model_path}")
            print("Using random moves instead...")
            self.model = None
    
    def get_move(self, game):
        """
        Get the AI's next move using the DQN model.
        
        Args:
            game: TicTacToeGame instance
            
        Returns:
            Tuple of (row, col) for the move
        """
        available_moves = game.get_available_moves()
        
        if not available_moves:
            return None
        
        # If model not loaded, use random move
        if self.model is None:
            return available_moves[np.random.randint(len(available_moves))]
        
        try:
            # Get Q-values for all possible moves
            state = game.board_to_state().reshape(1, -1)
            q_values = self.model.predict(state, verbose=0)[0]
            
            # Mask invalid moves with very low Q-values
            for i in range(9):
                if game.board.flatten()[i] != 0:
                    q_values[i] = -np.inf
            
            # Select move with highest Q-value
            best_move_idx = np.argmax(q_values)
            best_move = (best_move_idx // 3, best_move_idx % 3)
            
            # If the best move is invalid, use available moves
            if best_move not in available_moves:
                best_move = available_moves[np.argmax([q_values[m[0] * 3 + m[1]] 
                                                       for m in available_moves])]
            
            return best_move
        except Exception as e:
            print(f"Error in model prediction: {e}. Using random move...")
            return available_moves[np.random.randint(len(available_moves))]


def human_move(game):
    """
    Get a move from the human player.
    
    Args:
        game: TicTacToeGame instance
        
    Returns:
        Boolean indicating if move was successful
    """
    while True:
        try:
            position = input("Enter position (1-9) or row,col (0-2,0-2): ").strip()
            
            # Handle position input (1-9)
            if position.isdigit():
                pos = int(position)
                if 1 <= pos <= 9:
                    row, col = (pos - 1) // 3, (pos - 1) % 3
                else:
                    print("✗ Invalid position. Please enter 1-9 or row,col format.")
                    continue
            # Handle row,col format
            elif "," in position:
                parts = position.split(",")
                if len(parts) == 2:
                    try:
                        row, col = int(parts[0].strip()), int(parts[1].strip())
                    except ValueError:
                        print("✗ Invalid format. Please enter 1-9 or row,col (0-2,0-2).")
                        continue
                else:
                    print("✗ Invalid format. Please enter 1-9 or row,col (0-2,0-2).")
                    continue
            else:
                print("✗ Invalid input. Please enter 1-9 or row,col format.")
                continue
            
            # Attempt to make the move
            if game.make_move(row, col, 1):  # 1 represents human player
                return True
            else:
                print("✗ That position is already taken. Try again.")
        except Exception as e:
            print(f"✗ Error: {e}. Please try again.")


def play(model_path=None):
    """
    Main game loop for human vs AI Tic Tac Toe.
    
    Args:
        model_path: Optional path to the DQN model file
    """
    print("=" * 50)
    print("    TIC TAC TOE: Human vs AI (DQN)")
    print("=" * 50)
    print("\nControls:")
    print("  - Enter position 1-9 (top-left=1, bottom-right=9)")
    print("  - Or enter row,col (e.g., 0,0 for top-left)")
    print("\nYou are X, AI is O")
    print("=" * 50)
    
    # Initialize game and AI player
    game = TicTacToeGame()
    ai = DQNPlayer(model_path)
    
    game_count = 0
    human_wins = 0
    ai_wins = 0
    draws = 0
    
    while True:
        game.reset()
        game_count += 1
        print(f"\n{'=' * 50}")
        print(f"Game #{game_count}")
        print(f"Score - You: {human_wins} | AI: {ai_wins} | Draws: {draws}")
        print(f"{'=' * 50}")
        
        # Game loop
        while not game.game_over:
            game.display_board()
            
            # Human move
            print("Your turn (X):")
            human_move(game)
            game.display_board()
            
            # Check if human won
            winner = game.check_winner()
            if winner == 1:
                print("🎉 You won! Congratulations!")
                human_wins += 1
                game.game_over = True
                break
            elif winner == 0:
                print("🤝 It's a draw!")
                draws += 1
                game.game_over = True
                break
            
            # AI move
            print("AI's turn (O):")
            ai_move = ai.get_move(game)
            if ai_move:
                game.make_move(ai_move[0], ai_move[1], -1)  # -1 represents AI
                print(f"AI placed at position {ai_move[0] * 3 + ai_move[1] + 1}")
            
            game.display_board()
            
            # Check if AI won
            winner = game.check_winner()
            if winner == -1:
                print("🤖 AI won! Better luck next time.")
                ai_wins += 1
                game.game_over = True
            elif winner == 0:
                print("🤝 It's a draw!")
                draws += 1
                game.game_over = True
        
        # Ask to play again
        while True:
            play_again = input("\nPlay again? (yes/no): ").strip().lower()
            if play_again in ['yes', 'y']:
                break
            elif play_again in ['no', 'n']:
                print("\n" + "=" * 50)
                print("Final Score:")
                print(f"  You: {human_wins} wins")
                print(f"  AI:  {ai_wins} wins")
                print(f"  Draws: {draws}")
                print("=" * 50)
                print("Thanks for playing! 👋")
                return
            else:
                print("Please enter 'yes' or 'no'.")


if __name__ == "__main__":
    # Run the game
    play()
