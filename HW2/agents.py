import numpy as np
import random
import game

def print_INFO():
    """
    Prints your homework submission details.
    Please replace the placeholders (date, name, student ID) with valid information
    before submitting.
    """
    print(
        """========================================
        DATE: 2025/04/01
        STUDENT NAME: 黃奕淇
        STUDENT ID: 113550148
        ========================================
        """)


#
# Basic search functions: Minimax and Alpha‑Beta
#

def minimax(grid, depth, maximizingPlayer, dep=4):
    """
    TODO (Part 1): Implement recursive Minimax search for Connect Four.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
    """
    # base case
    if grid.terminate():
        return get_heuristic(grid), set()

    if depth == 0:
        # return game.get_heuristic(grid)
        return get_heuristic(grid), set()
        
    
    if depth > 0:
        candidates = set()
        # max value
        if maximizingPlayer == 1:
            v = -np.inf
            for col in grid.valid:
                val = minimax(game.drop_piece(grid, col), depth-1, not maximizingPlayer, dep)[0]
                if v < val:
                    v = val
                    candidates.clear()
                    candidates.add(col)
                elif v == val:
                    candidates.add(col)
            return v, candidates
        # min value
        else:
            v = np.inf
            for col in grid.valid:
                val = minimax(game.drop_piece(grid, col), depth-1, not maximizingPlayer, dep)[0]
                if v > val:
                    v = val
                    candidates.clear()
                    candidates.add(col)
                elif v == val:
                    candidates.add(col)
            return v, candidates

    return v, candidates


def alphabeta(grid, depth, maximizingPlayer, alpha, beta, dep=4):
    """
    TODO (Part 2): Implement Alpha-Beta pruning as an optimization to Minimax.

    Return:
      (boardValue, {setOfCandidateMoves})

    Where:
      - boardValue is the evaluated utility of the board state
      - {setOfCandidateMoves} is a set of columns that achieve this boardValue
      - Prune branches when alpha >= beta
    """
    # base case
    if grid.terminate():
        return get_heuristic(grid), set()

    if depth == 0:
        # return game.get_heuristic(grid)
        return get_heuristic(grid), set()
        
    
    if depth > 0:
        candidates = set()
        # max value
        if maximizingPlayer == 1:
            v = -np.inf
            for col in grid.valid:
                val = alphabeta(game.drop_piece(grid, col), depth-1, not maximizingPlayer, alpha, beta, dep)[0]
                if v < val:
                    v = val
                    candidates.clear()
                    candidates.add(col)
                elif v == val:
                    candidates.add(col)
                # pruning
                alpha = max(alpha, v)
                if alpha >= beta: return v, candidates
                
            return v, candidates
                
        # min value
        else:
            v = np.inf
            for col in grid.valid:
                val = alphabeta(game.drop_piece(grid, col), depth-1, not maximizingPlayer, alpha, beta, dep)[0]
                if v > val:
                    v = val
                    candidates.clear()
                    candidates.add(col)
                elif v == val:
                    candidates.add(col) 
                
                # pruning
                beta = min(beta, v)
                if beta <= alpha: return v, candidates
                
            return v, candidates
                

    return v, candidates


#
# Basic agents
#

def agent_minimax(grid):
    """
    Agent that uses the minimax() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(minimax(grid, 4, True)[1]))

def agent_alphabeta(grid):
    """
    Agent that uses the alphabeta() function with a default search depth (e.g., 4).
    Must return a single column (integer) where the piece is dropped.
    """
    return random.choice(list(alphabeta(grid, 4, True, -np.inf, np.inf)[1]))


def agent_reflex(grid):
    """
    A simple reflex agent provided as a baseline:
      - Checks if there's an immediate winning move.
      - Otherwise picks a random valid column.
    """
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    return random.choice(grid.valid)


def agent_strong(grid):
    """
    TODO (Part 3): Design your own agent (depth = 4) to consistently beat the Alpha-Beta agent (depth = 4).
    This agent will typically act as Player 2.
    """
    if grid.cnt == 2:
        # if grid.table[5][2] == 1:
        #     return 4
        # if grid.table[5][4] == 1:
        #     return 2
        return 3
    # if there is a winning move, do it
    wins = [c for c in grid.valid if game.check_winning_move(grid, c, grid.mark)]
    if wins:
        return random.choice(wins)
    # detect if opponent has a winning move
    for c in grid.valid:
        row = -1
        for r in range(grid.row-1, -1, -1):
            if grid.table[r][c] == 0:
                row = r
                break
        if row == -1:
            continue

        grid.table[r][c] = 1
        if grid.win(1):
            grid.table[r][c] = 0
            return c
        grid.table[r][c] = 0

    
    return random.choice(list(your_function(grid, 4, False, -np.inf, np.inf)[1]))


#
# Heuristic functions
#

def get_heuristic(board):
    """
    Evaluates the board from Player 1's perspective using a basic heuristic.

    Returns:
      - Large positive value if Player 1 is winning
      - Large negative value if Player 2 is winning
      - Intermediate scores based on partial connect patterns
    """
    num_twos       = game.count_windows(board, 2, 1)
    num_threes     = game.count_windows(board, 3, 1)
    num_twos_opp   = game.count_windows(board, 2, 2)
    num_threes_opp = game.count_windows(board, 3, 2)

    score = (
          1e10 * board.win(1)
        + 1e6  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e6  * num_threes_opp
        - 1e10 * board.win(2)
    )
    return score


def get_heuristic_strong(board, col):
    if board.win(1): return 1e10
    if board.win(2): return -1e10
    num_twos                                              = game.count_windows(board, 2, 1)
    multiattacks, potential_win, num_threes               = evaluate_attack_patterns(board, 3, 1)
    num_twos_opp                                          = game.count_windows(board, 2, 2)
    multiattacks_opp, potential_win_opp, num_threes_opp   = evaluate_attack_patterns(board, 3, 2)
    multiattacks_opp += two_three(board, 2)
    score = (
        + 1e7  * multiattacks
        + 1e5  * potential_win
        + 1e4  * num_threes
        + 10   * num_twos
        - 10   * num_twos_opp
        - 1e4  * num_threes_opp
        - 1e5  * potential_win_opp
        - 1e7  * multiattacks_opp
    )
    
    # Add center preference as a separate component with reasonable weight
    weighted = [1, 2, 10, 100, 10, 2, 1]
    center_preference = weighted[col] * 100

    return score - center_preference
    

def evaluate_attack_patterns(board, num_discs, piece):
    num_windows = 0
    # how many step it takes to reach to point and it is even or odd
    piece_choice = (piece+1)%2
    potential_to_win = 0
    # height of each column
    heights = height(board)
    # detect if there is a consecutive winning move in the same column
    attacks = np.asarray([[0] * board.column for _ in range(board.row)])
    multiwins = 0
    # Horizontal windows.
    for r in range(board.row):
        for c in range(board.column - (board.connect - 1)):
            window = list(board.table[r, c:c + board.connect])
            if game.check_window(board, window, num_discs, piece):
                num_windows += 1
                # 2 2 0 2
                if board.table[r][c] == piece and board.table[r][c+1] == piece and board.table[r][c+2] == 0:
                    h = heights[c+2]
                    attacks[r][c] = 1
                    if (h-r)%2 == piece_choice:
                        potential_to_win += 1
                # 2 0 2 2
                elif board.table[r][c] == piece and board.table[r][c+1] == 0 and board.table[r][c+2] == piece:
                    h = heights[c+1]
                    attacks[r][c] = 1
                    if (h-r)%2 == piece_choice:
                        potential_to_win += 1
                # 2 2 2
                elif board.table[r][c] == piece and board.table[r][c+1] == piece and board.table[r][c+2] == piece and c-1 in board.valid:
                    hl = heights[c-1]
                    hr = heights[c+3]
                    attacks[r][c-1] = 1 if hl < board.row else 0
                    attacks[r][c+3] = 1
                    if hl > r and (hl-r)%2 == piece_choice and (hr-r)%2 == piece_choice:
                        potential_to_win += 2
                    elif (hl > r and (hl-r)%2 == piece_choice) or ((hr-r)%2 == piece_choice):
                        potential_to_win += 1
                
    # Vertical windows.
    for r in range(board.row - (board.connect - 1)):
        for c in range(board.column):
            window = list(board.table[r:r + board.connect, c])
            if game.check_window(board, window, num_discs, piece) and c in board.valid:
                num_windows += 1
            
    # Positive diagonal windows.
    for r in range(board.row - (board.connect - 1)):
        for c in range(board.column - (board.connect - 1)):
            window = list(board.table[range(r, r + board.connect), range(c, c + board.connect)])
            if game.check_window(board, window, num_discs, piece):
                num_windows += 1
                # 2 2 0 2
                if board.table[r][c] == piece and board.table[r+1][c+1] == piece and board.table[r+2][c+2] == 0:
                    h = heights[c+2]
                    attacks[r+2][c+2] = 1
                    if (h-(r+2))%2 == piece_choice:
                        potential_to_win += 1
                # 2 0 2 2
                elif board.table[r][c] == piece and board.table[r+1][c+1] == 0 and board.table[r+2][c+2] == piece:
                    h = heights[c+1]
                    attacks[r+1][c+1] = 1
                    if (h-(r+1))%2 == piece_choice:
                        potential_to_win += 1
                # 2 2 2 0 
                elif board.table[r][c] == piece and board.table[r+1][c+1] == piece and board.table[r+2][c+2] == piece:
                    hl = heights[c-1]
                    hr = heights[c+3]       
                    1 
                    attacks[r+3][c+3] = 1
                    if c-1 in board.valid:
                        if r-1 < board.row: attacks[r-1][c-1] = 1
                        if hl > (r-1) and (hl-(r-1))%2 == piece_choice and (hr-(r+3))%2 == piece_choice:
                            potential_to_win += 2
                        elif (hl > (r-1) and (hl-(r-1))%2 == piece_choice) or ((hr-(r+3))%2 == piece_choice):
                            potential_to_win += 1
                    else:
                        if (hr-(r+3))%2 == piece_choice:
                            potential_to_win += 1
    # Negative diagonal windows.
    for r in range(board.connect - 1, board.row):
        for c in range(board.column - (board.connect - 1)):
            window = list(board.table[range(r, r - board.connect, -1), range(c, c + board.connect)])
            if game.check_window(board, window, num_discs, piece):
                num_windows += 1
                # 2 2 0 2
                if board.table[r][c] == piece and board.table[r-1][c+1] == piece and board.table[r-2][c+2] == 0:
                    h = heights[c+2]
                    attacks[r-2][c+2] = 1
                    if (h-(r-2))%2 == piece_choice:
                        potential_to_win += 1
                # 2 0 2 2
                elif board.table[r][c] == piece and board.table[r-1][c+1] == 0 and board.table[r-2][c+2] == piece:
                    h = heights[c+1]
                    attacks[r-1][c-1] = 1
                    if (h-(r-1))%2 == piece_choice:
                        potential_to_win += 1
                # 2 2 2 0 
                elif board.table[r][c] == piece and board.table[r-1][c+1] == piece and board.table[r-2][c+2] == piece:
                    hl = heights[c-1]
                    hr = heights[c+3]
                    attacks[r-3][c+3] = 1
                    if c-1 in board.valid:
                        if r+1 < board.row: attacks[r+1][c-1] = 1
                        if hl > (r+1) and (hl-(r+1))%2 == piece_choice and (hr-(r-3))%2 == piece_choice:
                            potential_to_win += 2
                        elif (hl > (r+1) and (hl-(r+1))%2 == piece_choice) or (hr > r and (hr-(r-3))%2 == piece_choice):
                            potential_to_win += 1
                    else:
                        if (hr-(r-3))%2 == piece_choice:
                            potential_to_win += 1
                
    
    for c in range(0, board.column):
        for r in range(0, board.row-1):
            if attacks[r][c] == 1 and attacks[r+1][c] == 1:
                multiwins += 1
    return multiwins, potential_to_win, num_windows

def two_three(grid, piece):
    wins = []
    if piece == 2:
        wins = [c for c in grid.valid if game.check_winning_move(grid, c, piece)]
    
    return len(wins) >= 2

def height(board):
    heights = []
    for c in range(0, board.column):
        r = 0
        while  r < board.row and board.table[r][c] == 0:
            r += 1
        heights.append(r)
    return heights

def your_function(grid, depth, maximizingPlayer, alpha, beta, col=0, dep=4):
    """
    A stronger search function that uses get_heuristic_strong() instead of get_heuristic().
    You can employ advanced features (e.g., improved move ordering, deeper lookahead).

    Return:
      (boardValue, {setOfCandidateMoves})

    Currently a placeholder returning (0, {0}).
    """
    # base case
    if grid.terminate():
        # return get_heuristic_AI(grid), set()
        return get_heuristic_strong(grid, col), set()
    if depth == 0:
        # return get_heuristic_AI(grid), set()
        return get_heuristic_strong(grid, col), set()
        
    
    if depth > 0:
        candidates = set()
        # max value
        if maximizingPlayer:
            v = -np.inf
            for col in grid.valid:
                val = your_function(game.drop_piece(grid, col), depth-1, not maximizingPlayer, alpha, beta, col, dep)[0]
                if v < val:
                    v = val
                    candidates.clear()
                    candidates.add(col)
                elif v == val:
                    candidates.add(col)
                # prunning
                # if v > beta: return v, candidates
                # alpha = max(alpha, v)
                
                
                alpha = max(alpha, v)
                if alpha > beta: return v, candidates

            return v, candidates   
                
                
        # min value
        else:
            v = np.inf
            for col in grid.valid:
                val = your_function(game.drop_piece(grid, col), depth-1, not maximizingPlayer, alpha, beta, col, dep)[0]
                if v > val:
                    v = val
                    candidates.clear()
                    candidates.add(col)
                elif v == val:
                    candidates.add(col) 
                # prunning
                # if alpha > v: return v, candidates
                # beta = min(beta, v)
                
                
                beta = min(v, beta)
                if alpha > beta: return v, candidates

            return v, candidates
        
    return v, candidates



