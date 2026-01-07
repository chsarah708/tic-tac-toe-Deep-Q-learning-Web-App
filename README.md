# Tic-Tac-Toe Deep Q-Learning Web App

## Introduction

Welcome to the **Tic-Tac-Toe Deep Q-Learning Web Application**! This project combines classic game development with cutting-edge machine learning techniques. It features an intelligent AI opponent trained using Deep Q-Learning (DQN), a powerful reinforcement learning algorithm. Challenge the AI and experience the convergence of game theory and neural networks in an interactive web environment.

## Features

### 🎮 Game Features
- **Interactive Tic-Tac-Toe Gameplay**: Play against an AI opponent with a clean, intuitive interface
- **AI-Powered Opponent**: Deep Q-Learning trained agent that learns optimal strategies through reinforcement learning
- **Real-Time Game State**: Dynamic board updates and instant move validation
- **Game History**: Track your wins, losses, and draws
- **Responsive Design**: Seamless experience across desktop and mobile devices

### 🤖 Machine Learning Features
- **Deep Q-Learning Algorithm**: State-of-the-art DQN implementation for game AI
- **Neural Network Integration**: Utilizes neural networks for Q-value approximation
- **Training Metrics**: Monitor training progress and agent performance
- **Reward System**: Custom reward structure optimized for tic-tac-toe strategy
- **Exploration vs Exploitation**: Balanced learning approach with epsilon-greedy strategy

### 💻 Technical Features
- **Web-Based Interface**: Modern, responsive web application
- **Real-Time Updates**: Smooth gameplay without page refreshes
- **Model Persistence**: Trained models saved for consistent performance
- **Error Handling**: Robust error management and user feedback

## Technologies

### Frontend
- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with flexbox and grid layouts
- **JavaScript (ES6+)**: Interactive game logic and UI management

### Backend
- **Python**: Core application logic and ML implementation
- **Flask/Django**: Web framework for server-side rendering
- **TensorFlow/Keras**: Deep learning framework for DQN implementation
- **NumPy**: Numerical computing and array operations

### Machine Learning
- **Deep Q-Learning (DQN)**: Reinforcement learning algorithm
- **Neural Networks**: Multi-layer perceptron for Q-value approximation
- **Experience Replay**: Batch learning from stored game experiences
- **Target Networks**: Improved stability in Q-value estimation

### DevOps & Tools
- **Git**: Version control
- **Docker** (optional): Containerization for deployment
- **Python Virtual Environment**: Dependency management

## Installation

### Prerequisites
- Python 3.8 or higher
- Node.js 14+ (if using Node-based frontend tooling)
- Git
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/chsarah708/tic-tac-toe-Deep-Q-learning-Web-App.git
cd tic-tac-toe-Deep-Q-learning-Web-App
```

### Step 2: Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Model (Optional)
If you want to train the Deep Q-Learning model from scratch:
```bash
python train_model.py --episodes 10000 --learning_rate 0.001
```

### Step 5: Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Playing the Game

1. **Start the Application**: Navigate to `http://localhost:5000` in your web browser
2. **Select Your Role**: Choose to play as X or O
3. **Make Your Move**: Click on an empty cell to place your mark
4. **AI Response**: The AI opponent automatically responds with its move
5. **Game Result**: The game concludes with win/loss/draw result

### Game Rules
- Standard tic-tac-toe rules apply (3x3 grid, three in a row wins)
- Player and AI alternate turns
- Game ends when someone wins or the board is full

### Training the Model

To train a new Deep Q-Learning model:

```bash
python train_model.py \
  --episodes 10000 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --epsilon_start 1.0 \
  --epsilon_end 0.01 \
  --epsilon_decay 0.995
```

**Parameters**:
- `episodes`: Number of training games (default: 10000)
- `batch_size`: Batch size for neural network training (default: 32)
- `learning_rate`: Learning rate for the optimizer (default: 0.001)
- `epsilon_start`: Initial exploration rate (default: 1.0)
- `epsilon_end`: Minimum exploration rate (default: 0.01)
- `epsilon_decay`: Decay rate for epsilon (default: 0.995)

### Evaluating Performance

```bash
python evaluate_model.py --games 100 --display_stats true
```

This will play 100 games and display win/loss/draw statistics.

## Project Structure

```
tic-tac-toe-Deep-Q-learning-Web-App/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── train_model.py                  # DQN training script
├── evaluate_model.py               # Model evaluation script
├── models/
│   ├── dqn_agent.py               # Deep Q-Learning agent implementation
│   ├── neural_network.py          # Neural network architecture
│   └── trained_model.h5           # Saved trained model
├── static/
│   ├── css/
│   │   └── style.css              # Styling
│   └── js/
│       └── game.js                # Frontend game logic
├── templates/
│   ├── index.html                 # Main game interface
│   └── stats.html                 # Statistics page
└── README.md                       # This file
```

## How Deep Q-Learning Works

### Algorithm Overview
The Deep Q-Learning algorithm learns optimal game strategies by:

1. **State Representation**: Board state encoded as neural network input
2. **Action Selection**: Agent selects moves using epsilon-greedy policy
3. **Experience Collection**: Games stored in replay memory
4. **Experience Replay**: Random batches used for training neural networks
5. **Q-Value Estimation**: Neural network approximates value of actions
6. **Target Network**: Separate network reduces overestimation bias

### Training Process
- Agent plays games against itself or random opponents
- Rewards given for wins (+1), losses (-1), and draws (0)
- Neural network learns to predict Q-values for state-action pairs
- Continuous improvement through repeated episodes

## Contributing

We welcome contributions to improve the project! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Commit: `git commit -m "Add detailed description of changes"`
5. Push: `git push origin feature/your-feature-name`
6. Submit a Pull Request

### Contribution Guidelines

#### Code Style
- Follow PEP 8 style guide for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Comment complex logic

#### Testing
- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

#### Documentation
- Update README.md for significant changes
- Add inline comments for complex algorithms
- Document new parameters and functions

#### Types of Contributions
- **Bug Fixes**: Report and fix bugs with detailed descriptions
- **Features**: Propose new game features or ML improvements
- **Performance**: Optimize training speed or inference time
- **Documentation**: Improve docs, add tutorials, fix typos
- **Testing**: Increase test coverage and reliability

### Reporting Issues
- Use descriptive titles
- Include steps to reproduce
- Provide error messages and logs
- Mention your environment (OS, Python version, etc.)

### Pull Request Process
1. Update documentation for any new features
2. Add tests for new functionality
3. Ensure CI/CD pipeline passes
4. Request review from maintainers
5. Address feedback promptly

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow/Keras team for excellent ML frameworks
- Reinforcement learning community for DQN inspiration
- Flask framework for web development
- Contributors and testers who help improve the project

## Contact & Support

- **Author**: Sarah (@chsarah708)
- **GitHub Repository**: [tic-tac-toe-Deep-Q-learning-Web-App](https://github.com/chsarah708/tic-tac-toe-Deep-Q-learning-Web-App)
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join conversations in GitHub Discussions

---

**Happy Playing and Learning!** 🎯🤖

*Last Updated: January 7, 2026*
