# Tic-Tac-Toe Deep Q-Learning Web App

A web application that implements a Tic-Tac-Toe game powered by Deep Q-Learning (DQN) artificial intelligence. Play against an AI opponent trained using reinforcement learning.

## Project Structure

```
tic-tac-toe-Deep-Q-learning-Web-App/
├── .gitignore
├── LICENSE
├── README.md
├── __pycache__/
├── .venv/
├── dqn_model.pth
├── game_logic.py
├── play.py
├── requirements.txt
├── server.py
├── train.py
└── static/
    ├── index.html
    ├── script.js
    └── style.css
```

## Project Files

- **game_logic.py**: Core game logic for Tic-Tac-Toe, including board state management and win condition checking
- **train.py**: Deep Q-Learning training script to train the AI model
- **play.py**: Script to play against the trained AI model
- **server.py**: Flask/web server to run the web application
- **dqn_model.pth**: Trained DQN model weights
- **requirements.txt**: Python dependencies
- **static/**: Frontend files
  - **index.html**: Main HTML structure for the game interface
  - **script.js**: JavaScript logic for game interaction and API communication
  - **style.css**: Styling for the game interface

## Features

- **Deep Q-Learning AI**: An intelligent opponent trained using reinforcement learning
- **Web-Based Interface**: Play the game directly in your browser
- **Real-time Gameplay**: Smooth interaction between player and AI
- **Training Capability**: Includes scripts to train and improve the AI model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/chsarah708/tic-tac-toe-Deep-Q-learning-Web-App.git
cd tic-tac-toe-Deep-Q-learning-Web-App
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Playing the Game

Run the web server:
```bash
python server.py
```

Then open your browser and navigate to `http://localhost:5000` to play the game.

### Training the Model

To train a new AI model:
```bash
python train.py
```

This will generate or update the `dqn_model.pth` file with the trained model weights.

### Testing Against AI

To test the game logic and AI:
```bash
python play.py
```

## Requirements

See `requirements.txt` for all Python dependencies. Key packages include:
- Flask (for web server)
- PyTorch (for deep learning)
- NumPy (for numerical operations)

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## Author

Created by chsarah708

## Acknowledgments

This project combines game development with reinforcement learning to create an intelligent Tic-Tac-Toe opponent using Deep Q-Networks (DQN).
