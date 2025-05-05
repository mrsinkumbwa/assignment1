import copy

# Initialize empty board
initial_state = [
    [None, None, None],
    [None, None, None],
    [None, None, None]
]

def player(board):
    """Determine current player"""
    x = sum(row.count('X') for row in board)
    o = sum(row.count('O') for row in board)
    return 'X' if x == o else 'O'

def actions(board):
    """Get available moves"""
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] is None]

def result(board, action):
    """Create new board state"""
    i, j = action
    if board[i][j] is not None:
        raise ValueError("Invalid move")
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board

def winner(board):
    """Check for winner"""
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0]: return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i]: return board[0][i]
    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0]: return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2]: return board[0][2]
    return None

def terminal(board):
    """Check game end"""
    return winner(board) or all(cell is not None for row in board for cell in row)

def utility(board):
    """Calculate game outcome"""
    win = winner(board)
    return 1 if win == 'X' else -1 if win == 'O' else 0

def minimax(board):
    """Optimal move calculation"""
    if terminal(board): return None
    
    def max_val(board):
        if terminal(board): return utility(board), None
        value = -float('inf')
        move = None
        for action in actions(board):
            new_val, _ = min_val(result(board, action))
            if new_val > value:
                value, move = new_val, action
        return value, move
    
    def min_val(board):
        if terminal(board): return utility(board), None
        value = float('inf')
        move = None
        for action in actions(board):
            new_val, _ = max_val(result(board, action))
            if new_val < value:
                value, move = new_val, action
        return value, move
    
    return max_val(board)[1] if player(board) == 'X' else min_val(board)[1]
def print_board(board):
    """Display the game board"""
    symbols = {None: ' ', 'X': 'X', 'O': 'O'}
    for i, row in enumerate(board):
        print(f" {symbols[row[0]]} | {symbols[row[1]]} | {symbols[row[2]]} ")
        if i < 2: print("-----------")

def play_game():
    """Interactive game loop"""
    board = copy.deepcopy(initial_state)
    human = 'X'  # Human plays X
    ai = 'O'     # AI plays O
    
    print("TIC-TAC-TOE\nHuman (X) vs AI (O)\n")
    
    while not terminal(board):
        print_board(board)
        current = player(board)
        
        if current == human:
            print("\nYour turn (X)")
            while True:
                try:
                    row = int(input("Row (0-2): "))
                    col = int(input("Column (0-2): "))
                    if (row, col) in actions(board):
                        break
                    print("Invalid move! Try again.")
                except ValueError:
                    print("Numbers 0-2 only!")
            board = result(board, (row, col))
        else:
            print("\nAI's turn (O)...")
            move = minimax(board)
            board = result(board, move)
    
    print("\nFinal board:")
    print_board(board)
    win = winner(board)
    print(f"\n{'You won!' if win == human else 'AI won!' if win else "It's a tie!"}")

# Start the game
play_game()