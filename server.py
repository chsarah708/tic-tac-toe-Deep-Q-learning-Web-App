from flask import Flask, jsonify, request, send_from_directory
import os
import json
from datetime import datetime

app = Flask(__name__, static_folder='static', static_url_path='')

# Store game state in memory (in production, use database)
game_state = {
    'board': [' ' for _ in range(9)],
    'current_player': 'X',
    'game_over': False,
    'winner': None,
    'move_count': 0
}


def check_winner(board):
    """Check if there's a winner on the board."""
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]               # diagonals
    ]
    
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != ' ':
            return board[combo[0]]
    
    return None


def is_board_full(board):
    """Check if the board is full (draw)."""
    return ' ' not in board


def reset_game():
    """Reset game state to initial conditions."""
    global game_state
    game_state = {
        'board': [' ' for _ in range(9)],
        'current_player': 'X',
        'game_over': False,
        'winner': None,
        'move_count': 0
    }


@app.route('/')
def index():
    """Serve the main index.html file."""
    return send_from_directory('static', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, etc.)."""
    return send_from_directory('static', filename)


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """
    Initialize a new game.
    
    Returns:
        JSON with new game state
    """
    reset_game()
    return jsonify({
        'status': 'success',
        'message': 'New game started',
        'game_state': game_state,
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/api/player_move', methods=['POST'])
def player_move():
    """
    Process a player move.
    
    Expected JSON body:
        {
            'position': <int 0-8>,
            'player': <str 'X' or 'O'>
        }
    
    Returns:
        JSON with updated game state or error message
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'position' not in data or 'player' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields: position and player'
            }), 400
        
        position = int(data['position'])
        player = data['player'].upper()
        
        # Validate position
        if position < 0 or position > 8:
            return jsonify({
                'status': 'error',
                'message': 'Invalid position. Must be between 0 and 8.'
            }), 400
        
        # Validate player
        if player not in ['X', 'O']:
            return jsonify({
                'status': 'error',
                'message': 'Invalid player. Must be X or O.'
            }), 400
        
        # Check if game is already over
        if game_state['game_over']:
            return jsonify({
                'status': 'error',
                'message': 'Game is already over. Start a new game.'
            }), 409
        
        # Check if position is already occupied
        if game_state['board'][position] != ' ':
            return jsonify({
                'status': 'error',
                'message': f'Position {position} is already occupied.'
            }), 409
        
        # Check if it's the correct player's turn
        if player != game_state['current_player']:
            return jsonify({
                'status': 'error',
                'message': f'It is {game_state["current_player"]}\'s turn, not {player}\'s.'
            }), 409
        
        # Make the move
        game_state['board'][position] = player
        game_state['move_count'] += 1
        
        # Check for winner
        winner = check_winner(game_state['board'])
        if winner:
            game_state['winner'] = winner
            game_state['game_over'] = True
            return jsonify({
                'status': 'success',
                'message': f'Player {winner} wins!',
                'game_state': game_state,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
        # Check for draw
        if is_board_full(game_state['board']):
            game_state['game_over'] = True
            return jsonify({
                'status': 'success',
                'message': 'Game is a draw!',
                'game_state': game_state,
                'timestamp': datetime.utcnow().isoformat()
            }), 200
        
        # Switch player
        game_state['current_player'] = 'O' if player == 'X' else 'X'
        
        return jsonify({
            'status': 'success',
            'message': f'Move accepted. Next player: {game_state["current_player"]}',
            'game_state': game_state,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid position value. Must be an integer.'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        }), 500


@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    """
    Retrieve the current game state.
    
    Returns:
        JSON with current game state
    """
    return jsonify({
        'status': 'success',
        'game_state': game_state,
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors (method not allowed)."""
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405


if __name__ == '__main__':
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
