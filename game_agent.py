"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    raise NotImplementedError


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        # print(legal_moves, 'legal moves passed in')
        if len(legal_moves) == 0:
            return
        move = legal_moves[0];

        def search(depth, game):
            if self.method == "minimax":
                score, move = self.minimax(game, depth)
            elif self.method == "alphabeta":
                score, move = self.alphabeta(game, depth)
            else:
                print("method chosen isn't minimax or alphabeta")
            return score, move

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative == True:
                depth=1
                while True:
                    score, move = search(depth, game)
                    depth += depth
            else:
                score, move = search(self.search_depth, game)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return move

        # Return the best move from the last completed search iteration
        print("move to return {}, with score {}".format(move, score))
        return move


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!

        current_player = game.active_player

        # base case for recursion
        if depth<=0:
            # print(game.get_player_location(current_player), current_player)
            # Should get the current location of --us--, and the score
            # for that location given the game board
            current_move = game.get_player_location(self)
            return self.score(game, self), current_move

        # initialize
        legal_moves = game.get_legal_moves();
        choosen_move = legal_moves[0]
        move_score = -math.inf if maximizing_player==True else math.inf

        # print(legal_moves, "legal moves")
        for move in legal_moves:
            temp_board = game.forecast_move(move)
            # Recursive step
            temp_score, _= self.minimax(temp_board, depth-1, not maximizing_player)
            # print("temp_move {}, temp_score {}, depth {}".format(_, temp_score, depth))
            # check whether we're maximizing or minimizing score
            if maximizing_player == True:
                if temp_score > move_score:
                    choosen_move, move_score = move, temp_score
            else:
                if temp_score < move_score:
                    choosen_move, move_score = move, temp_score

        # print("Best move: {}, move_score {}, depth {}".format(choosen_move, move_score, depth))

        return move_score, choosen_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        current_player = game.active_player

        # base case for recursion
        if depth<=0:
            # print(game.get_player_location(current_player), current_player)
            # Should get the current location of --us--, and the score
            # for that location given the game board
            current_move = game.get_player_location(self)
            return self.score(game, self), current_move

        # initialize
        legal_moves = game.get_legal_moves();
        choosen_move = legal_moves[0]
        move_score = -math.inf if maximizing_player==True else math.inf

        # print(legal_moves, "legal moves")
        for move in legal_moves:
            temp_board = game.forecast_move(move)
            # Recursive step
            temp_score, _= self.alphabeta(temp_board, depth-1, alpha=alpha, beta=beta, maximizing_player=(not maximizing_player))
            # print("ALPHABETA: temp_move {}, temp_score {}, depth {}, maximizing_player {}".format(_, temp_score, depth-1, maximizing_player))
            # print("alpha {}, beta {}".format(alpha, beta))
            # check whether we're maximizing or minimizing score
            if maximizing_player == True:
                if temp_score > move_score:
                    choosen_move, move_score = move, temp_score
                if beta <= move_score:
                    return move_score, choosen_move
                    # print("Beta exceeded with {} greater than beta score {}".format(move_score, beta))
                alpha = max(alpha, move_score)
                # print("alpha",alpha)
            else:
                if temp_score < move_score:
                    choosen_move, move_score = move, temp_score
                if alpha >= move_score:
                    return move_score, choosen_move
                    # print("Alpha SHORTCUTTED {} less than alpha score {}".format(move_score, alpha))
                beta = min(beta, move_score)

        # print("ALPHABETA NOT SHORTCUTTED: Best move: {}, move_score {}, depth {}".format(choosen_move, move_score, depth))

        return move_score, choosen_move
