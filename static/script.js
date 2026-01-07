// ============================================================================
// Tic-Tac-Toe Deep Q-Learning Web App - Main Script
// Game State Management, API Calls, Game Logic, UI Updates, and More
// ============================================================================

// ============================================================================
// GAME STATE MANAGEMENT
// ============================================================================

const GameState = {
  board: Array(9).fill(null),
  currentPlayer: 'X',
  gameActive: true,
  gameMode: 'ai', // 'ai' or 'pvp'
  moves: [],
  gameStartTime: null,
  aiDifficulty: 'hard', // 'easy', 'medium', 'hard'
  
  // Initialize game
  reset() {
    this.board = Array(9).fill(null);
    this.currentPlayer = 'X';
    this.gameActive = true;
    this.moves = [];
    this.gameStartTime = Date.now();
  },
  
  // Make a move
  makeMove(index) {
    if (!this.gameActive || this.board[index] !== null) {
      return false;
    }
    this.board[index] = this.currentPlayer;
    this.moves.push({ index, player: this.currentPlayer, timestamp: Date.now() });
    return true;
  },
  
  // Switch player
  switchPlayer() {
    this.currentPlayer = this.currentPlayer === 'X' ? 'O' : 'X';
  },
  
  // Get board copy
  getBoardCopy() {
    return [...this.board];
  },
  
  // Check winner
  getWinner() {
    const lines = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [0, 3, 6],
      [1, 4, 7],
      [2, 5, 8],
      [0, 4, 8],
      [2, 4, 6]
    ];
    
    for (let line of lines) {
      const [a, b, c] = line;
      if (this.board[a] && this.board[a] === this.board[b] && this.board[a] === this.board[c]) {
        return { winner: this.board[a], line };
      }
    }
    return null;
  },
  
  // Check if board is full
  isFull() {
    return this.board.every(cell => cell !== null);
  },
  
  // Get available moves
  getAvailableMoves() {
    return this.board
      .map((cell, index) => cell === null ? index : null)
      .filter(index => index !== null);
  }
};

// ============================================================================
// SCORE AND STATISTICS MANAGEMENT
// ============================================================================

const ScoreManager = {
  storageKey: 'ticTacToeScores',
  
  // Get all scores
  getScores() {
    const scores = localStorage.getItem(this.storageKey);
    return scores ? JSON.parse(scores) : this.getDefaultScores();
  },
  
  // Get default scores
  getDefaultScores() {
    return {
      totalGames: 0,
      playerWins: 0,
      aiWins: 0,
      draws: 0,
      pvpGames: 0,
      streak: 0,
      bestStreak: 0,
      lastGameResult: null,
      lastGameDate: null,
      gamesPlayedToday: 0
    };
  },
  
  // Save scores
  saveScores(scores) {
    localStorage.setItem(this.storageKey, JSON.stringify(scores));
  },
  
  // Add win for player
  addPlayerWin() {
    const scores = this.getScores();
    scores.totalGames++;
    scores.playerWins++;
    scores.streak++;
    scores.bestStreak = Math.max(scores.streak, scores.bestStreak);
    scores.lastGameResult = 'win';
    scores.lastGameDate = new Date().toISOString();
    scores.gamesPlayedToday = this.getGamesPlayedToday() + 1;
    this.saveScores(scores);
    return scores;
  },
  
  // Add win for AI
  addAIWin() {
    const scores = this.getScores();
    scores.totalGames++;
    scores.aiWins++;
    scores.streak = 0;
    scores.lastGameResult = 'loss';
    scores.lastGameDate = new Date().toISOString();
    scores.gamesPlayedToday = this.getGamesPlayedToday() + 1;
    this.saveScores(scores);
    return scores;
  },
  
  // Add draw
  addDraw() {
    const scores = this.getScores();
    scores.totalGames++;
    scores.draws++;
    scores.lastGameResult = 'draw';
    scores.lastGameDate = new Date().toISOString();
    scores.gamesPlayedToday = this.getGamesPlayedToday() + 1;
    this.saveScores(scores);
    return scores;
  },
  
  // Get win rate
  getWinRate() {
    const scores = this.getScores();
    if (scores.totalGames === 0) return 0;
    return ((scores.playerWins / scores.totalGames) * 100).toFixed(2);
  },
  
  // Get games played today
  getGamesPlayedToday() {
    const scores = this.getScores();
    if (!scores.lastGameDate) return 0;
    const lastGameDate = new Date(scores.lastGameDate);
    const today = new Date();
    if (lastGameDate.toDateString() === today.toDateString()) {
      return scores.gamesPlayedToday;
    }
    return 0;
  },
  
  // Reset all scores
  resetScores() {
    localStorage.removeItem(this.storageKey);
  }
};

// ============================================================================
// API CALLS
// ============================================================================

const APIClient = {
  baseURL: '/api',
  
  // Get AI move
  async getAIMove(board, difficulty = 'hard') {
    try {
      const response = await fetch(`${this.baseURL}/ai-move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ board, difficulty })
      });
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      return data.move;
    } catch (error) {
      console.error('Error getting AI move:', error);
      return this.getRandomMove(board);
    }
  },
  
  // Get random move (fallback)
  getRandomMove(board) {
    const availableMoves = GameState.getAvailableMoves();
    if (availableMoves.length === 0) return -1;
    return availableMoves[Math.floor(Math.random() * availableMoves.length)];
  },
  
  // Train AI with game data
  async trainAI(gameData) {
    try {
      const response = await fetch(`${this.baseURL}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gameData)
      });
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error training AI:', error);
      return null;
    }
  },
  
  // Get game statistics
  async getGameStats() {
    try {
      const response = await fetch(`${this.baseURL}/stats`);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching game stats:', error);
      return null;
    }
  },
  
  // Reset AI model
  async resetAI() {
    try {
      const response = await fetch(`${this.baseURL}/reset`, { method: 'POST' });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('Error resetting AI:', error);
      return null;
    }
  }
};

// ============================================================================
// GAME FUNCTIONS
// ============================================================================

const GameFunctions = {
  // Initialize game
  init() {
    GameState.reset();
    UIManager.updateBoard();
    UIManager.updateStatus();
    UIManager.updateScoreboard();
  },
  
  // Handle player move
  playerMove(index) {
    if (!GameState.gameActive || GameState.board[index] !== null) {
      return;
    }
    
    if (!GameState.makeMove(index)) {
      return;
    }
    
    UIManager.updateBoard();
    
    const result = GameState.getWinner();
    if (result) {
      this.endGame(GameState.currentPlayer, result.line);
      return;
    }
    
    if (GameState.isFull()) {
      this.endGame('draw');
      return;
    }
    
    GameState.switchPlayer();
    UIManager.updateStatus();
    
    // AI turn if in AI mode
    if (GameState.gameMode === 'ai' && GameState.currentPlayer === 'O') {
      setTimeout(() => this.aiMove(), 500);
    }
  },
  
  // Handle AI move
  async aiMove() {
    if (!GameState.gameActive || GameState.currentPlayer !== 'O') {
      return;
    }
    
    const aiMoveIndex = await APIClient.getAIMove(
      GameState.board,
      GameState.aiDifficulty
    );
    
    if (aiMoveIndex === -1 || aiMoveIndex === null) {
      this.endGame('draw');
      return;
    }
    
    if (!GameState.makeMove(aiMoveIndex)) {
      this.aiMove();
      return;
    }
    
    UIManager.updateBoard();
    
    const result = GameState.getWinner();
    if (result) {
      this.endGame(GameState.currentPlayer, result.line);
      return;
    }
    
    if (GameState.isFull()) {
      this.endGame('draw');
      return;
    }
    
    GameState.switchPlayer();
    UIManager.updateStatus();
  },
  
  // End game
  endGame(winner, winningLine = null) {
    GameState.gameActive = false;
    
    let message = '';
    if (winner === 'draw') {
      message = "It's a Draw!";
      ScoreManager.addDraw();
    } else if (winner === 'X') {
      message = 'You Win! 🎉';
      ScoreManager.addPlayerWin();
    } else if (winner === 'O') {
      message = 'AI Wins! 🤖';
      ScoreManager.addAIWin();
    }
    
    UIManager.updateStatus(message);
    UIManager.updateScoreboard();
    
    if (winningLine) {
      UIManager.highlightWinningLine(winningLine);
    }
    
    // Train AI if player won
    if (winner === 'X' && GameState.gameMode === 'ai') {
      this.trainAIWithGameData();
    }
    
    UIManager.showGameOverModal(message);
  },
  
  // Train AI with game data
  async trainAIWithGameData() {
    const gameData = {
      board: GameState.board,
      moves: GameState.moves,
      result: 'loss',
      duration: Date.now() - GameState.gameStartTime,
      difficulty: GameState.aiDifficulty
    };
    
    const result = await APIClient.trainAI(gameData);
    if (result) {
      console.log('AI trained with game data:', result);
    }
  },
  
  // Set game mode
  setGameMode(mode) {
    GameState.gameMode = mode;
    this.init();
  },
  
  // Set AI difficulty
  setAIDifficulty(difficulty) {
    GameState.aiDifficulty = difficulty;
  },
  
  // Undo last move
  undoLastMove() {
    if (GameState.moves.length < 1) return;
    
    const lastMove = GameState.moves.pop();
    GameState.board[lastMove.index] = null;
    GameState.currentPlayer = lastMove.player;
    GameState.gameActive = true;
    
    UIManager.updateBoard();
    UIManager.updateStatus();
  },
  
  // Get game duration
  getGameDuration() {
    if (!GameState.gameStartTime) return 0;
    return Math.floor((Date.now() - GameState.gameStartTime) / 1000);
  }
};

// ============================================================================
// UI MANAGER
// ============================================================================

const UIManager = {
  // Update board display
  updateBoard() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach((cell, index) => {
      const value = GameState.board[index];
      cell.textContent = value || '';
      cell.className = `cell ${value ? value.toLowerCase() : ''}`.trim();
      cell.dataset.index = index;
    });
  },
  
  // Update game status
  updateStatus(message = null) {
    const statusElement = document.getElementById('status');
    if (!statusElement) return;
    
    if (message) {
      statusElement.textContent = message;
    } else if (!GameState.gameActive) {
      statusElement.textContent = 'Game Over!';
    } else {
      const player = GameState.currentPlayer === 'X' ? 'You' : 'AI';
      statusElement.textContent = `${player}'s Turn`;
    }
  },
  
  // Update scoreboard
  updateScoreboard() {
    const scores = ScoreManager.getScores();
    
    const elements = {
      totalGames: document.getElementById('total-games'),
      playerWins: document.getElementById('player-wins'),
      aiWins: document.getElementById('ai-wins'),
      draws: document.getElementById('draws'),
      winRate: document.getElementById('win-rate'),
      streak: document.getElementById('streak'),
      bestStreak: document.getElementById('best-streak')
    };
    
    if (elements.totalGames) elements.totalGames.textContent = scores.totalGames;
    if (elements.playerWins) elements.playerWins.textContent = scores.playerWins;
    if (elements.aiWins) elements.aiWins.textContent = scores.aiWins;
    if (elements.draws) elements.draws.textContent = scores.draws;
    if (elements.winRate) elements.winRate.textContent = ScoreManager.getWinRate() + '%';
    if (elements.streak) elements.streak.textContent = scores.streak;
    if (elements.bestStreak) elements.bestStreak.textContent = scores.bestStreak;
  },
  
  // Highlight winning line
  highlightWinningLine(winningLine) {
    const cells = document.querySelectorAll('.cell');
    winningLine.forEach(index => {
      cells[index].classList.add('winner');
    });
  },
  
  // Clear winning line highlight
  clearWinningLine() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => cell.classList.remove('winner'));
  },
  
  // Show game over modal
  showGameOverModal(message) {
    const modal = document.getElementById('game-over-modal');
    if (!modal) return;
    
    const modalMessage = modal.querySelector('.modal-message');
    if (modalMessage) {
      modalMessage.textContent = message;
    }
    
    this.showModal('game-over-modal');
  },
  
  // Show modal
  showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    modal.style.display = 'flex';
    modal.classList.add('active');
  },
  
  // Close modal
  closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    modal.style.display = 'none';
    modal.classList.remove('active');
  },
  
  // Show notification
  showNotification(message, type = 'info') {
    const container = document.getElementById('notification-container');
    if (!container) return;
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    container.appendChild(notification);
    
    setTimeout(() => {
      notification.classList.add('fade-out');
      setTimeout(() => notification.remove(), 300);
    }, 3000);
  },
  
  // Update difficulty display
  updateDifficultyDisplay(difficulty) {
    const buttons = document.querySelectorAll('[data-difficulty]');
    buttons.forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.difficulty === difficulty) {
        btn.classList.add('active');
      }
    });
  },
  
  // Update game mode display
  updateGameModeDisplay(mode) {
    const buttons = document.querySelectorAll('[data-mode]');
    buttons.forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.mode === mode) {
        btn.classList.add('active');
      }
    });
  },
  
  // Disable board
  disableBoard() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => cell.style.pointerEvents = 'none');
  },
  
  // Enable board
  enableBoard() {
    const cells = document.querySelectorAll('.cell');
    cells.forEach(cell => cell.style.pointerEvents = 'auto');
  }
};

// ============================================================================
// MODAL FUNCTIONS
// ============================================================================

const ModalManager = {
  // Initialize modals
  init() {
    this.setupModalClosers();
    this.setupModalBackdrops();
  },
  
  // Setup modal close buttons
  setupModalClosers() {
    const closeButtons = document.querySelectorAll('.modal-close');
    closeButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const modal = e.target.closest('.modal');
        if (modal) {
          UIManager.closeModal(modal.id);
        }
      });
    });
  },
  
  // Setup modal backdrops
  setupModalBackdrops() {
    const modals = document.querySelectorAll('.modal');
    modals.forEach(modal => {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) {
          UIManager.closeModal(modal.id);
        }
      });
    });
  },
  
  // Open settings modal
  openSettings() {
    UIManager.showModal('settings-modal');
  },
  
  // Open statistics modal
  openStatistics() {
    UIManager.showModal('statistics-modal');
    this.updateStatisticsModal();
  },
  
  // Update statistics modal
  updateStatisticsModal() {
    const scores = ScoreManager.getScores();
    const statsElements = {
      totalGames: document.getElementById('modal-total-games'),
      playerWins: document.getElementById('modal-player-wins'),
      aiWins: document.getElementById('modal-ai-wins'),
      draws: document.getElementById('modal-draws'),
      winRate: document.getElementById('modal-win-rate'),
      currentStreak: document.getElementById('modal-current-streak'),
      bestStreak: document.getElementById('modal-best-streak'),
      lastGame: document.getElementById('modal-last-game')
    };
    
    if (statsElements.totalGames) statsElements.totalGames.textContent = scores.totalGames;
    if (statsElements.playerWins) statsElements.playerWins.textContent = scores.playerWins;
    if (statsElements.aiWins) statsElements.aiWins.textContent = scores.aiWins;
    if (statsElements.draws) statsElements.draws.textContent = scores.draws;
    if (statsElements.winRate) statsElements.winRate.textContent = ScoreManager.getWinRate() + '%';
    if (statsElements.currentStreak) statsElements.currentStreak.textContent = scores.streak;
    if (statsElements.bestStreak) statsElements.bestStreak.textContent = scores.bestStreak;
    if (statsElements.lastGame) {
      const lastDate = scores.lastGameDate ? new Date(scores.lastGameDate).toLocaleDateString() : 'Never';
      statsElements.lastGame.textContent = lastDate;
    }
  },
  
  // Open help modal
  openHelp() {
    UIManager.showModal('help-modal');
  },
  
  // Confirm reset scores
  confirmResetScores() {
    if (confirm('Are you sure you want to reset all scores? This cannot be undone.')) {
      ScoreManager.resetScores();
      UIManager.updateScoreboard();
      UIManager.showNotification('Scores reset successfully', 'success');
      this.updateStatisticsModal();
    }
  }
};

// ============================================================================
// EVENT LISTENERS
// ============================================================================

const EventListeners = {
  // Initialize all event listeners
  init() {
    this.setupBoardListeners();
    this.setupButtonListeners();
    this.setupGameModeListeners();
    this.setupDifficultyListeners();
    this.setupModalListeners();
    this.setupKeyboardShortcuts();
    ModalManager.init();
  },
  
  // Board cell click listeners
  setupBoardListeners() {
    const board = document.getElementById('board');
    if (!board) return;
    
    board.addEventListener('click', (e) => {
      const cell = e.target.closest('.cell');
      if (cell) {
        const index = parseInt(cell.dataset.index);
        GameFunctions.playerMove(index);
      }
    });
  },
  
  // Main button listeners
  setupButtonListeners() {
    const newGameBtn = document.getElementById('new-game-btn');
    if (newGameBtn) {
      newGameBtn.addEventListener('click', () => {
        UIManager.clearWinningLine();
        GameFunctions.init();
        UIManager.showNotification('New game started', 'info');
      });
    }
    
    const undoBtn = document.getElementById('undo-btn');
    if (undoBtn) {
      undoBtn.addEventListener('click', () => {
        GameFunctions.undoLastMove();
        UIManager.showNotification('Move undone', 'info');
      });
    }
    
    const settingsBtn = document.getElementById('settings-btn');
    if (settingsBtn) {
      settingsBtn.addEventListener('click', () => ModalManager.openSettings());
    }
    
    const statsBtn = document.getElementById('stats-btn');
    if (statsBtn) {
      statsBtn.addEventListener('click', () => ModalManager.openStatistics());
    }
    
    const helpBtn = document.getElementById('help-btn');
    if (helpBtn) {
      helpBtn.addEventListener('click', () => ModalManager.openHelp());
    }
    
    const resetScoresBtn = document.getElementById('reset-scores-btn');
    if (resetScoresBtn) {
      resetScoresBtn.addEventListener('click', () => ModalManager.confirmResetScores());
    }
    
    const playAgainBtn = document.getElementById('play-again-btn');
    if (playAgainBtn) {
      playAgainBtn.addEventListener('click', () => {
        UIManager.closeModal('game-over-modal');
        UIManager.clearWinningLine();
        GameFunctions.init();
      });
    }
  },
  
  // Game mode listeners
  setupGameModeListeners() {
    const modeButtons = document.querySelectorAll('[data-mode]');
    modeButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        GameState.gameMode = mode;
        GameFunctions.init();
        UIManager.updateGameModeDisplay(mode);
        UIManager.showNotification(`Game mode: ${mode.toUpperCase()}`, 'info');
      });
    });
  },
  
  // Difficulty listeners
  setupDifficultyListeners() {
    const difficultyButtons = document.querySelectorAll('[data-difficulty]');
    difficultyButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const difficulty = btn.dataset.difficulty;
        GameFunctions.setAIDifficulty(difficulty);
        UIManager.updateDifficultyDisplay(difficulty);
        UIManager.showNotification(`Difficulty: ${difficulty}`, 'info');
      });
    });
  },
  
  // Modal button listeners
  setupModalListeners() {
    const modalButtons = document.querySelectorAll('[data-modal]');
    modalButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const modalId = btn.dataset.modal;
        UIManager.showModal(modalId);
      });
    });
  }
};

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

const KeyboardShortcuts = {
  // Initialize keyboard shortcuts
  init() {
    document.addEventListener('keydown', (e) => {
      this.handleKeyPress(e);
    });
  },
  
  // Handle key press
  handleKeyPress(e) {
    const key = e.key.toLowerCase();
    
    // Number keys (1-9) for board positions
    if (key >= '1' && key <= '9') {
      const index = parseInt(key) - 1;
      if (index < 9 && GameState.gameActive) {
        GameFunctions.playerMove(index);
      }
    }
    
    // Spacebar for new game
    if (e.code === 'Space') {
      e.preventDefault();
      UIManager.clearWinningLine();
      GameFunctions.init();
      UIManager.showNotification('New game started', 'info');
    }
    
    // Ctrl+Z for undo
    if ((e.ctrlKey || e.metaKey) && key === 'z') {
      e.preventDefault();
      GameFunctions.undoLastMove();
      UIManager.showNotification('Move undone', 'info');
    }
    
    // 'S' for settings
    if (key === 's' && !this.isInputFocused()) {
      e.preventDefault();
      ModalManager.openSettings();
    }
    
    // 'T' for statistics
    if (key === 't' && !this.isInputFocused()) {
      e.preventDefault();
      ModalManager.openStatistics();
    }
    
    // 'H' for help
    if (key === 'h' && !this.isInputFocused()) {
      e.preventDefault();
      ModalManager.openHelp();
    }
    
    // Escape to close modals
    if (key === 'escape') {
      const modals = document.querySelectorAll('.modal.active');
      modals.forEach(modal => UIManager.closeModal(modal.id));
    }
  },
  
  // Check if input is focused
  isInputFocused() {
    const activeElement = document.activeElement;
    return activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA';
  }
};

// ============================================================================
// LOCAL STORAGE PERSISTENCE
// ============================================================================

const StorageManager = {
  // Game settings key
  settingsKey: 'ticTacToeSettings',
  
  // Get all settings
  getSettings() {
    const settings = localStorage.getItem(this.settingsKey);
    return settings ? JSON.parse(settings) : this.getDefaultSettings();
  },
  
  // Get default settings
  getDefaultSettings() {
    return {
      gameMode: 'ai',
      aiDifficulty: 'hard',
      soundEnabled: true,
      theme: 'light',
      notifications: true
    };
  },
  
  // Save settings
  saveSettings(settings) {
    localStorage.setItem(this.settingsKey, JSON.stringify(settings));
  },
  
  // Update specific setting
  updateSetting(key, value) {
    const settings = this.getSettings();
    settings[key] = value;
    this.saveSettings(settings);
  },
  
  // Load settings on app start
  loadSettings() {
    const settings = this.getSettings();
    GameState.gameMode = settings.gameMode;
    GameState.aiDifficulty = settings.aiDifficulty;
    return settings;
  },
  
  // Clear all data
  clearAllData() {
    localStorage.removeItem(this.settingsKey);
    ScoreManager.resetScores();
  }
};

// ============================================================================
// INITIALIZATION
// ============================================================================

const App = {
  // Initialize the entire app
  init() {
    console.log('Initializing Tic-Tac-Toe Game...');
    
    // Load settings
    const settings = StorageManager.loadSettings();
    
    // Initialize game
    GameFunctions.init();
    
    // Setup event listeners
    EventListeners.init();
    
    // Initialize keyboard shortcuts
    KeyboardShortcuts.init();
    
    // Update display
    UIManager.updateDifficultyDisplay(GameState.aiDifficulty);
    UIManager.updateGameModeDisplay(GameState.gameMode);
    UIManager.updateScoreboard();
    
    console.log('Game initialized successfully');
  }
};

// Start app when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => App.init());
} else {
  App.init();
}

// ============================================================================
// EXPORT FOR TESTING (if using modules)
// ============================================================================

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    GameState,
    ScoreManager,
    APIClient,
    GameFunctions,
    UIManager,
    ModalManager,
    EventListeners,
    KeyboardShortcuts,
    StorageManager,
    App
  };
}
