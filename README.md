# Tic-Tac-Toe Deep Q-Learning Web App

## Introduction

This project implements an intelligent Tic-Tac-Toe game powered by Deep Q-Learning (DQL), a reinforcement learning technique. The application provides a web-based interface where users can play against an AI opponent trained using deep neural networks. The AI learns optimal strategies through self-play and experience, making it a formidable opponent that improves its decision-making over time.

This project demonstrates the practical application of machine learning in game AI, combining reinforcement learning principles with modern web technologies.

## Features

- **AI Opponent Powered by Deep Q-Learning**: Play against an intelligent agent trained with deep reinforcement learning
- **Interactive Web Interface**: User-friendly web application built with modern frontend technologies
- **Real-time Game State Management**: Dynamic board updates and move validation
- **Training Dashboard**: Monitor the training progress and AI learning metrics
- **Multiple Difficulty Levels**: Adjust the game difficulty based on your skill level
- **Game History**: Track previous games and review past moves
- **Responsive Design**: Play on desktop, tablet, or mobile devices
- **Model Persistence**: Save and load trained models for consistent AI performance

## Technologies

### Backend
- **Python 3.8+**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for building neural networks
- **Flask/Django**: Web framework for API endpoints
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis

### Frontend
- **React/Vue.js**: JavaScript framework for interactive UI
- **HTML5 & CSS3**: Structure and styling
- **Axios/Fetch API**: HTTP requests to backend
- **Canvas/SVG**: Game board visualization

### DevOps & Deployment
- **Docker**: Containerization for consistent environments
- **Git**: Version control
- **pytest**: Testing framework

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 14+ (for frontend)
- Git
- Virtual environment manager (venv or conda)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/chsarah708/tic-tac-toe-Deep-Q-learning-Web-App.git
   cd tic-tac-toe-Deep-Q-learning-Web-App
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained model (optional)**
   ```bash
   python scripts/download_model.py
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Create .env file for API configuration**
   ```
   REACT_APP_API_URL=http://localhost:5000/api
   ```

## Usage

### Running the Application

1. **Start the backend server**
   ```bash
   python app.py
   ```
   The Flask server will run on `http://localhost:5000`

2. **Start the frontend development server** (in a new terminal)
   ```bash
   cd frontend
   npm start
   ```
   The React app will open at `http://localhost:3000`

3. **Access the application**
   Open your browser and navigate to `http://localhost:3000`

### Playing the Game

1. **Start a new game**: Click "New Game" button
2. **Make your move**: Click on an empty cell (you are X, AI is O)
3. **AI responds**: The AI automatically calculates and makes its move
4. **Win conditions**: Get three in a row (horizontal, vertical, or diagonal) to win
5. **View stats**: Check your game statistics and win rate in the dashboard

### Training the Model

To train a new model from scratch:

```bash
python train_model.py --episodes 10000 --learning_rate 0.001 --batch_size 64
```

**Training parameters:**
- `--episodes`: Number of training episodes (default: 10000)
- `--learning_rate`: DQL learning rate (default: 0.001)
- `--batch_size`: Batch size for training (default: 64)
- `--save_interval`: Save model every N episodes (default: 100)

## Code Overview

### Project Structure

```
tic-tac-toe-Deep-Q-learning-Web-App/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── config.py              # Configuration settings
│   ├── requirements.txt        # Python dependencies
│   ├── models/
│   │   ├── dql_agent.py       # Deep Q-Learning agent implementation
│   │   ├── neural_network.py  # Neural network architecture
│   │   └── game_state.py      # Game state representation
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── validators.py      # Input validation
│   ├── utils/
│   │   ├── game_logic.py      # Tic-tac-toe game rules
│   │   └── helpers.py         # Utility functions
│   └── trained_models/        # Saved model weights
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── services/          # API service calls
│   │   └── App.js             # Main app component
│   ├── package.json           # npm dependencies
│   └── .env                   # Environment variables
├��─ scripts/
│   ├── train_model.py         # Model training script
│   ├── evaluate_model.py      # Model evaluation script
│   └── download_model.py      # Download pre-trained weights
├── tests/
│   ├── test_game_logic.py     # Game logic tests
│   ├── test_dql_agent.py      # Agent tests
│   └── test_api.py            # API endpoint tests
├── docker-compose.yml         # Docker composition
├── Dockerfile                 # Docker configuration
└── README.md                  # This file
```

### Key Components

#### 1. **Deep Q-Learning Agent** (`models/dql_agent.py`)
- Implements the DQL algorithm with experience replay
- Manages exploration-exploitation trade-off
- Maintains Q-value estimates for game states

#### 2. **Neural Network** (`models/neural_network.py`)
- Multi-layer perceptron for Q-value approximation
- Input: Game board state (flattened 9-element vector)
- Output: Q-values for each possible action (9 actions)
- Architecture: 9 → 128 → 64 → 9 (with ReLU activation)

#### 3. **Game Logic** (`utils/game_logic.py`)
- Validates moves and enforces game rules
- Detects win/loss/draw conditions
- Manages game state transitions

#### 4. **API Routes** (`api/routes.py`)
- `/api/game/new` - Start new game
- `/api/game/move` - Submit player move
- `/api/game/state` - Get current game state
- `/api/stats` - Retrieve player statistics
- `/api/ai/difficulty` - Set AI difficulty level

## Definitions

### Deep Q-Learning (DQL)
An off-policy reinforcement learning algorithm that uses a deep neural network to approximate Q-values. Q-values represent the expected future reward for taking a specific action in a given state.

**Key concepts:**
- **Q-value**: Estimate of total future reward from a state-action pair
- **Reward**: Immediate feedback from the environment (+1 for win, -1 for loss, 0 for draw)
- **Experience Replay**: Store past transitions in memory and train on random batches to improve stability
- **Target Network**: Separate network used for stability during training

### Exploration vs. Exploitation
- **Exploration**: Randomly choosing actions to discover new strategies
- **Exploitation**: Using learned knowledge to select best-known actions
- **Epsilon-Greedy**: Strategy that explores with probability ε and exploits with probability 1-ε

### Game State Representation
The 3x3 Tic-Tac-Toe board is represented as a 9-element vector:
```
[0, 1, 2]
[3, 4, 5]  → [state[0], state[1], state[2], state[3], ...]
[6, 7, 8]
```
Values: 0 (empty), 1 (player X), -1 (player O)

### Reward Structure
- **+1**: Agent wins
- **-1**: Agent loses
- **0**: Game ends in draw
- **0**: Game continues (intermediate states)

## Contributing Guidelines

We welcome contributions! Please follow these guidelines:

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
   ```bash
   git clone https://github.com/your-username/tic-tac-toe-Deep-Q-learning-Web-App.git
   ```
3. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Workflow

1. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Make your changes**
   - Follow PEP 8 code style guidelines
   - Write clear, descriptive commit messages
   - Add comments for complex logic

3. **Write/update tests**
   ```bash
   pytest tests/ -v
   ```
   - Maintain or improve code coverage
   - Test edge cases and error handling

4. **Format and lint your code**
   ```bash
   black backend/
   flake8 backend/
   pylint backend/
   ```

### Commit Message Guidelines

Use clear, descriptive commit messages:
```
[Type] Brief description

Detailed explanation if necessary

- Bullet point for changes
- Another change
```

**Types**: feat (feature), fix (bug fix), docs (documentation), style (formatting), refactor, test, chore

Example:
```
[feat] Add difficulty level selector to UI

- Add difficulty level buttons to game interface
- Implement difficulty parameter in DQL agent
- Update API to handle difficulty settings
```

### Pull Request Process

1. **Update your branch** with the latest main
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Push your changes** to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to related issues (if any)
   - Screenshots/GIFs for UI changes

4. **Respond to feedback** and make requested changes

5. **Wait for approval** before merging

### Code Review Process

All contributions require code review. Reviewers will check:
- Code quality and style consistency
- Test coverage and completeness
- Documentation accuracy
- Performance implications
- Security concerns

### Areas for Contribution

- **Bug fixes**: Report and fix issues
- **Features**: Add new game modes, AI improvements, UI enhancements
- **Documentation**: Improve README, add tutorials, write docstrings
- **Testing**: Increase test coverage, add integration tests
- **Performance**: Optimize neural network, improve game speed
- **UI/UX**: Enhance user interface and experience

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Report inappropriate behavior

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or suggestions:
- **Open an issue** on GitHub for bug reports
- **Start a discussion** for feature requests
- **Check existing issues** before creating duplicates

## Acknowledgments

- TensorFlow/Keras community for excellent deep learning tools
- Open source community for inspiration and support
- Contributors who have helped improve this project

---

**Last Updated**: January 7, 2026

For more information, visit the [project repository](https://github.com/chsarah708/tic-tac-toe-Deep-Q-learning-Web-App)
