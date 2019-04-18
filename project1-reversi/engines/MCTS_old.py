import time
from engines import Engine
from random import choice
from copy import deepcopy
from math import log, sqrt


class Node(object):
    """
    This class represents the tree node in a Monte-Carlo tree.
    """
    def __init__(self, board, color, depth, parent=None, premove=None):
        """
        :param board: the current board
        :param color: the current moving color
        :param depth: depth of the node in the search tree
        :param parent: the parent node
        :param premove: the last move
        """
        self.board = board
        self.color = color
        self.depth = depth
        self.parent = parent
        self.premove = premove
        # Number of visits
        self.visit_num = 0
        # Valid moves
        self.remain_valid_moves = board.get_legal_moves(color)
        # Children of current node
        self.children = []
        # N: Visit times
        self.N = 0
        # Q: Total reward, stands for the number of white wins
        self.Q = 0

    def add_child(self, board, new_move):
        # Append a new node to the current one
        self.remain_valid_moves.remove(new_move)

        new_board = deepcopy(board)
        new_board.execute_move(new_move, self.color)
        new_node = Node(new_board, -self.color, self.depth + 1, self, new_move)
        self.children.append(new_node)

        return new_node

    def is_terminal(self):
        # Check if the node is a terminal one
        return not self.board.get_legal_moves(self.color)

    def is_fully_expanded(self):
        # Check if the node is fully expanded
        if not self.remain_valid_moves:
            return True
        else:
            return False
    

class MctsEngine(Engine):
    """
    This game engine uses UCT search method to look for the best
    move of a given condition.
    """
    def __init__(self):
        self.move_num = 0
        self.comp_budget = 5
        self.root = None
        # Parameter for MCTS
        self.Cp = 1

    def get_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        # Get the root for the search tree
        self.root = Node(board, color, 1)
        # Get the move number
        self.move_num = move_num

        # Get the total time
        total_time = self.comp_budget
        while total_time > 0:
            start_time = time.time()
            
            # Get the node to expand
            new_node = self.tree_policy(self.root)
            # expand_time = time.time()
            # print("tree_policy time: ", expand_time - start_time)

            # Expand the node and get the reward
            reward = self.default_policy(new_node)
            # simulate_time = time.time()
            # print("default_policy time: ", simulate_time - expand_time)

            # Use reward to change N and Q value of previous nodes
            self.backup(new_node, reward)
            # backup_time = time.time()
            # print("backup time: ", backup_time - simulate_time)

            end_time = time.time()
            interval = end_time - start_time
            
            # print(interval)
            # print(reward)
            # print(total_time)
            # print()
            total_time -= interval

        # Return bestchild
        v, node = self.best_child(self.root)
        return node.premove
    
    def tree_policy(self, node):
        # Look for a new node to expand
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                v, node = self.best_child(node)
        return node

    def is_terminal(self, board):
        # Check if the board is a terminal one
        return not board.get_legal_moves(1) and not board.get_legal_moves(-1)

    def terminal_reward(self, board):
        # white wins -> 1
        # black wins -> 0
        # tie -> 0.5
        reward = 0
        for line in board.pieces:
            for entry in line:
                reward += entry
        if reward > 0:
            return 1
        elif reward < 0:
            return 0
        else:
            return 0.5

    def default_policy(self, node):
        # Get the reward of the given node
        cur_reward = 0
        cur_color = node.color
        cur_move_num = self.move_num
        cur_board = deepcopy(node.board)
        remaining_num = 5

        while remaining_num > 0:
            while cur_move_num < 64:
                legal_set = cur_board.get_legal_moves(cur_color)
                if not legal_set:
                    cur_move_num += 1
                    cur_color *= -1
                    continue
                new_move = choice(legal_set)
                cur_board.execute_move(new_move, cur_color)
                cur_color *= -1
                cur_move_num += 1

                # Check if the current board is a terminal one
                if self.is_terminal(cur_board):
                    cur_reward += self.terminal_reward(cur_board)
                    break

            # Reset the game
            cur_color = node.color
            cur_move_num = self.move_num
            cur_board = deepcopy(node.board)
            remaining_num -= 1
        
        # Return the final reward
        return cur_reward

    def backup(self, node, reward):
        # Use reward to change N and Q value of previous nodes
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent

    def expand(self, node):
        # Expand the node to get an unvisited node
        moves = node.remain_valid_moves
        move = choice(moves)
        new_node = node.add_child(node.board, move)
        return new_node

    def best_child(self, node):
        # Look for the node with biggest reward
        if node.color == 1:
            # Color is white
            child_value = [(child.Q/child.N + self.Cp*sqrt(2*log(node.N)/child.N))
                            for child in node.children]
        elif node.color == -1:
            # Color is black
            child_value = [(1 - child.Q/child.N + self.Cp*sqrt(2*log(node.N)/child.N))
                            for child in node.children]
        else:
            return 0, None

        value = max(child_value)
        idx = child_value.index(value)
        return value, node.children[idx]


engine = MctsEngine
