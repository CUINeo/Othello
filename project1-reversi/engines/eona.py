from engines import Engine
from copy import deepcopy
from random import shuffle

# Version 11.25
class EonaEngine(Engine):
    def __init__(self):
        # initialize a few pre-calculated tables
        self.alpha_beta = False
        fill_bit_table()
        fill_lsb_table()
        fill_radial_map()
        self.depth = 5
        # timing history
        self.last_time_remaining = 0

#         self.nodes = 0
#         self.duplicate = {}
#         self.parent = 0
#         self.children = 0

    def get_move(self, board, color, move_num=None,
                 time_remaining=None, time_opponent=None):
        # simple opening book
        if move_num == 0:
            if color < 0:
                return (2, 4)
            else:
                if board[2][4] == -1 or board[3][5] == -1:
                    return (2, 5)
                elif board[4][2] == -1 or board[5][3] == -1:
                    return (5, 2)

        # timing
        mvs_remain = 32.0 - move_num
        alloc_time = time_remaining / (mvs_remain if mvs_remain != 0 else 1)
        last_time = self.last_time_remaining - time_remaining

        if move_num > 5 and time_remaining > 10:
            if alloc_time >= 0.6 and move_num < 12 or alloc_time > 0.8 and move_num < 18 or alloc_time > 1.5:
                self.depth = 6
            else:
                self.depth = 5
        if last_time > 0.6 * time_remaining:
            if time_remaining < 7:
                self.depth = 4
            else:
                self.depth = 5

        if last_time > 2 * alloc_time:
            self.depth = 5

        if time_remaining < 3:
            self.depth = 4
        if time_remaining < 0.8:
            self.depth = 3
        if time_remaining < 0.5:
            self.depth = 2
        if time_remaining < 0.3:
            self.depth = 1
#         print "alloc", alloc_time, "last", last_time, "self.depth", self.depth, "at round", move_num, "time remain", time_remaining
        self.last_time_remaining = time_remaining

        W, B = to_bitboard(board)

        wb = (W, B) if color > 0 else (B, W)

        if self.alpha_beta:
            res = self.alphabeta(wb[0], wb[1], self.depth, -float("inf"), float("inf"))
        else:
            res = self.minimax(wb[0], wb[1], 3)
        return to_move(res[1])

        # debugging
#         return self._debug_bb(board, color, DEPTH)[1]

        # Get a list of all legal moves.
#         if self.alpha_beta:
#             return self.alphabeta(board, color, DEPTH, -float("inf"), float("inf"))[1]
#         else:
#             return self.minimax(board, color, DEPTH)[1]

    def minimax(self, W, B, depth):
#         self.nodes += 1
#         if (W, B) in self.duplicate:
#             self.duplicate[W, B] = True
#         else:
#             self.duplicate[W, B] = False

        if depth == 0:
            return (self.eval(W, B), None)

        movemap = move_gen(W, B)
#         self.parent += 1
#         self.children += count_bit(movemap)

        best = - float("inf")
        bestmv = None
        if movemap != 0:
            bestmv, movemap = pop_lsb(movemap)
        else:
            return (best, None)

        mv = bestmv
        while True:
            tmpW = W
            tmpB = B
            flipmask = flip(W, B, mv)
            tmpW ^= flipmask | BIT[mv]
            tmpB ^= flipmask

            score = -self.minimax(tmpB, tmpW, depth - 1)[0]
            if score > best:
                best = score
                bestmv = mv

            if movemap == 0:
                break
            else:
                mv, movemap = pop_lsb(movemap)

        #print "color", "white" if color == 1 else "black", "depth", depth, "best", best, "legals", len(movelist)
        return (best, bestmv)

    def alphabeta(self, W, B, depth, alpha, beta):
        movemap = move_gen(W, B)
        best = alpha

        mvlist = []
        while movemap != 0:
            mv, movemap = pop_lsb(movemap)
            mvlist.append(mv)

        if len(mvlist) == 0:
            # we don't have any legal moves. Let's see if opponent has any
            if move_gen(B, W) != 0:
                if depth == 1:
                    return (- self.eval(B, W), None)
                else:
                    return  (-self.alphabeta(B, W, depth - 1, -beta, -best)[0], None)
            else:
                # no one has legal move. Game ends
                wscore = count_bit(W)
                bscore = count_bit(B)
                return (1e7 if wscore > bscore else \
                            0 if wscore == bscore else \
                            -1e7, None)
        elif len(mvlist) == 1 and depth == self.depth:
            # if there's one move at the top level, don't search at all
            return (0, mvlist[0])
        else:
            shuffle(mvlist)
            bestmv = mvlist[0]

        for mv in mvlist:
            tmpW = W
            tmpB = B
            flipmask = flip(W, B, mv)
            tmpW ^= flipmask | BIT[mv]
            tmpB ^= flipmask

            if depth == 1:
                score = - self.eval(tmpB, tmpW)
            else: # save a recursive call
                score = - self.alphabeta(tmpB, tmpW, depth - 1, -beta, -best)[0]

            # principal variation search
#             res = self.alphabeta(tmpB, tmpW, depth-1, -(best+1), -best)
#             score = -res[0]
#             if score > best and score < beta:
#                 score = - self.alphabeta(tmpB, tmpW, depth - 1, -beta, -best)[0]
#             else:
#                 return (score, res[1])

            if score > best:
                best = score
                bestmv = mv

            if best >= beta:
                return (best, bestmv)

        return (best, bestmv)

    WEIGHTS = \
    [-3, -7, 11, -4, 8, 1, 2]
    P_RINGS = [0x4281001818008142,
               0x42000000004200,
               0x2400810000810024,
               0x24420000422400,
               0x1800008181000018,
               0x18004242001800,
               0x3C24243C0000]
    P_CORNER = 0x8100000000000081
    P_SUB_CORNER = 0x42C300000000C342

    def eval(self, W, B):
        w0 = W & BIT[0] != 0
        w1 = W & BIT[7] != 0
        w2 = W & BIT[56] != 0
        w3 = W & BIT[63] != 0
        b0 = B & BIT[0] != 0
        b1 = B & BIT[7] != 0
        b2 = B & BIT[56] != 0
        b3 = B & BIT[63] != 0

        # stability
        wunstable = bunstable = 0
        if w0 != 1 and b0 != 1:
            wunstable += (W & BIT[1] != 0) + (W & BIT[8] != 0) + (W & BIT[9] != 0)
            bunstable += (B & BIT[1] != 0) + (B & BIT[8] != 0) + (B & BIT[9] != 0)
        if w1 != 1 and b1 != 1:
            wunstable += (W & BIT[6] != 0) + (W & BIT[14] != 0) + (W & BIT[15] != 0)
            bunstable += (B & BIT[6] != 0) + (B & BIT[14] != 0) + (B & BIT[15] != 0)
        if w2 != 1 and b2 != 1:
            wunstable += (W & BIT[48] != 0) + (W & BIT[49] != 0) + (W & BIT[57] != 0)
            bunstable += (B & BIT[48] != 0) + (B & BIT[49] != 0) + (B & BIT[57] != 0)
        if w3 != 1 and b3 != 1:
            wunstable += (W & BIT[62] != 0) + (W & BIT[54] != 0) + (W & BIT[55] != 0)
            bunstable += (B & BIT[62] != 0) + (B & BIT[54] != 0) + (B & BIT[55] != 0)

        scoreunstable = - 10.0 * (wunstable - bunstable)

        # piece difference
        wpiece = (w0 + w1 + w2 + w3) * 100.0
        for i in range(len(self.WEIGHTS)):
            wpiece += self.WEIGHTS[i] * count_bit(W & self.P_RINGS[i])
        bpiece = (b0 + b1 + b2 + b3) * 100.0
        for i in range(len(self.WEIGHTS)):
            bpiece += self.WEIGHTS[i] * count_bit(B & self.P_RINGS[i])

#         scorepiece = \
#             10.0 * wpiece / (wpiece + bpiece) if wpiece > bpiece \
#             else -10.0 * bpiece / (wpiece + bpiece) if wpiece < bpiece \
#             else 0
        scorepiece = wpiece - bpiece

        # mobility
        wmob = count_bit(move_gen(W, B))

        scoremob = 5 * wmob

        return scorepiece + scoreunstable + scoremob


    def minimax_old(self, board, color, depth):
        if depth == 0:
            return (self.eval_old(board, color), None)
        movelist = board.get_legal_moves(color)
        best = - float("inf")
        bestmv = None if len(movelist)==0 else movelist[0]
        for mv in movelist:
            newboard = deepcopy(board)
            newboard.execute_move(mv, color)
            res = self.minimax_old(newboard, color * -1, depth - 1)
            score = - res[0]
            if score > best:
                best = score
                bestmv = mv
        #print "color", "white" if color == 1 else "black", "depth", depth, "best", best, "legals", len(movelist)
        return (best, bestmv)

    def alphabeta_old(self, board, color, depth, alpha, beta):
        if depth == 0:
            return (self.eval_old(board, color), None)
        movelist = board.get_legal_moves(color)
        best = alpha
        bestmv = None if len(movelist)==0 else movelist[0]
        for mv in movelist:
            newboard = deepcopy(board)
            newboard.execute_move(mv, color)
            res = self.alphabeta_old(newboard, color * -1, depth - 1, -beta, -best)
            score = - res[0]
            if score > best:
                best = score
                bestmv = mv
            if best >= beta:
                return (best, bestmv)
        return (best, bestmv)

    def _get_cost(self, board, color, move): return 0

    def eval_old(self, board, color):
        # Count the # of pieces of each color on the board
        num_pieces_op = len(board.get_squares(color*-1))
        num_pieces_me = len(board.get_squares(color))
        return num_pieces_me - num_pieces_op

    #------------- DEBUG ONLY ------------
    def _debug_bb(self, board, color, depth):
        if depth == 0:
            return (self.eval_old(board, color), None)
        movelist = board.get_legal_moves(color)

        W, B = to_bitboard(board)
        movelistw = sorted([to_bitmove(m) for m in board.get_legal_moves(1)])
        movelistb = sorted([to_bitmove(m) for m in board.get_legal_moves(-1)])
        movemapw = move_gen(W, B)
        movemapb = move_gen(B, W)
        movemapw_ = movemapw
        movemapb_ = movemapb
        w_count = count_bit(movemapw)
        b_count = count_bit(movemapb)

        try:
            assert w_count == len(movelistw)
            assert b_count == len(movelistb)
            i = 0
            while movemapw != 0:
                m, movemapw = pop_lsb(movemapw)
                assert movelistw[i] == m
                i += 1
            assert w_count == i
            i = 0
            while movemapb != 0:
                m, movemapb = pop_lsb(movemapb)
                assert movelistb[i] == m
                i += 1
            assert b_count == i
        except (IndexError, AssertionError) as e:
            print("MOVEGEN CRASH DEBUG")
            print("white movelist")
            print(movelistw)
            print(movemapw_)
            print_bitboard(movemapw_)
            print("wcount =", w_count)
            print("black movelist")
            print(movelistb)
            print(movemapb_)
            print_bitboard(movemapb_)
            print("bcount =", b_count)
            print(e)
            raise

        best = - float("inf")
        bestmv = None if len(movelist)==0 else movelist[0]
        for mv in movelist:
            newboard = deepcopy(board)
            newboard.execute_move(mv, color)

            ww, bb = to_bitboard(newboard)
            tmpW = W
            tmpB = B
            mvtmp = to_bitmove(mv)
            if color > 0:
                flipmask = flip(W, B, mvtmp)
                tmpW ^= flipmask | BIT[mvtmp]
                tmpB ^= flipmask
            else:
                flipmask = flip(B, W, mvtmp)
                tmpB ^= flipmask | BIT[mvtmp]
                tmpW ^= flipmask

            try:
                assert ww == tmpW
                assert bb == tmpB
            except AssertionError:
                print("MAKE MOVE CRASH DEBUG")
                board.display([1,2,3])
                print("move")
                print_bitboard(BIT[mvtmp])
                print("FLIP")
                print_bitboard(flipmask)
                print("CORRECT W")
                print_bitboard(ww)
                print_bitboard(tmpW)
                print("CORRECT B")
                print_bitboard(bb)
                print_bitboard(tmpB)
                raise AssertionError
            res = self._debug_bb(newboard, color * -1, depth - 1)
            score = - res[0]
            if score > best:
                best = score
                bestmv = mv
        return (best, bestmv)

######-------- bitboard representation -------
def fill_bit_table():
    global BIT
    BIT = [1 << n for n in range(64)]

def move_gen_sub(P, mask, dir):
    dir2 = int(dir * 2)
    flip1  = mask & (P << dir)
    flip2  = mask & (P >> dir)
    flip1 |= mask & (flip1 << dir)
    flip2 |= mask & (flip2 >> dir)
    mask1  = mask & (mask << dir)
    mask2  = mask & (mask >> dir)
    flip1 |= mask1 & (flip1 << dir2)
    flip2 |= mask2 & (flip2 >> dir2)
    flip1 |= mask1 & (flip1 << dir2)
    flip2 |= mask2 & (flip2 >> dir2)
    return (flip1 << dir) | (flip2 >> dir)

def move_gen(P, O):
    mask = O & 0x7E7E7E7E7E7E7E7E
    return ((move_gen_sub(P, mask, 1) \
            | move_gen_sub(P, O, 8)  \
            | move_gen_sub(P, mask, 7) \
            | move_gen_sub(P, mask, 9)) & ~(P|O)) & FULL_MASK

def print_bitboard(BB):
    bitarr = [1 if (1<<i) & BB != 0 else 0 for i in range(64)]
    s = ""
    for rk in range(7, -1, -1):
        for fl in range(8):
            s += str(bitarr[fl + 8 * rk]) + " "
        s += "\n"
    print(s)

def to_bitboard(board):
    W = 0
    B = 0
    for r in range(8):
        for c in range(8):
            if board[c][r] == -1:
                B |= BIT[8 * r + c]
            elif board[c][r] == 1:
                W |= BIT[8 * r + c]
    return (W, B)

def to_move(bitmove):
    return (bitmove % 8, bitmove // 8)

def to_bitmove(move):
    return move[0] + 8 * move[1]

RADIAL_MAP = {}

def fill_radial_map():
    rad_map = {-1: (-1, 0), 1:(1, 0), -8:(0, -1), 8:(0, 1), -7:(1, -1), 7:(-1, 1), -9:(-1, -1), 9:(1, 1)}
    for dir, dirtup in list(rad_map.items()):
        lis = [0] * 64
        for sqr in range(64):
            mask = 0
            sq = sqr
            x, y = to_move(sq)
            sq += dir
            x += dirtup[0]
            y += dirtup[1]
            while 0 <= x < 8 and 0 <= y < 8 and 0 <= sq < 64:
                mask |= BIT[sq]
                sq += dir
                x += dirtup[0]
                y += dirtup[1]
            lis[sqr] = mask
        RADIAL_MAP[dir] = lis

DIR = [
[1, -7, -8],
[-1,-9,-8],
[1,8,9],
[7,8,-1],
[8,9,1,-7,-8],
[-1,1,-7,-8,-9],
[7,8,-1,-9,-8],
[7,8,9,-1,1],
[-1, 1, -7,7,-8,8,-9,9]]

SQ_DIR = \
[2, 2, 7, 7, 7, 7, 3, 3,
 2, 2, 7, 7, 7, 7, 3, 3 ,
 4, 4, 8, 8, 8, 8, 6, 6,
 4, 4, 8, 8, 8, 8, 6, 6,
 4, 4, 8, 8, 8, 8, 6, 6,
 4, 4, 8, 8, 8, 8, 6, 6,
 0, 0, 5, 5, 5, 5, 1, 1,
 0, 0, 5, 5, 5, 5, 1, 1 ]

def flip(W, B, mv):
    mask = 0
    for dir in DIR[SQ_DIR[mv]]:
        mvtmp = mv
        mvtmp += dir
        while mvtmp >= 0 and mvtmp < 64 and (BIT[mvtmp] & B != 0) and (BIT[mvtmp] & RADIAL_MAP[dir][mv] != 0):
            mvtmp += dir
        if (mvtmp >= 0 and mvtmp < 64 and BIT[mvtmp] & W != 0) and (BIT[mvtmp] & RADIAL_MAP[dir][mv] != 0):
            mvtmp -= dir
            while mvtmp != mv:
                mask |= BIT[mvtmp]
                mvtmp -= dir

    return mask

FULL_MASK = 0xFFFFFFFFFFFFFFFF
LSB_HASH = 0x07EDD5E59A4E28C2
def fill_lsb_table():
    bitmap = 1
    global LSB_TABLE
    LSB_TABLE = [0] * 64
    for i in range(64):
        LSB_TABLE[(((bitmap & (~bitmap + 1)) * LSB_HASH) & FULL_MASK) >> 58] = i
        bitmap <<= 1

def lsb(bitmap):
    return LSB_TABLE[(((bitmap & (~bitmap + 1)) * LSB_HASH) & FULL_MASK) >> 58]

def pop_lsb(bitmap):
    l= lsb(bitmap)
    bitmap &= bitmap-1
    return l, bitmap & FULL_MASK

def count_bit(b):
    b -=  (b >> 1) & 0x5555555555555555;
    b  = (((b >> 2) & 0x3333333333333333) + (b & 0x3333333333333333));
    b  = ((b >> 4) + b)  & 0x0F0F0F0F0F0F0F0F;
    return ((b * 0x0101010101010101) & FULL_MASK) >> 56

def count_bit_2(b):
    raise DeprecationWarning
    cnt = 0
    for i in range(64):
        if b & BIT[i] != 0:
            cnt += 1
    return cnt

# helper: print a hex representation of the position
def pos2hex(*plist):
    n = 0
    for p in plist:
        n |= BIT[p]
    print_bitboard(n)
    s = hex(n).upper()
    return "0x" + s[2:]

engine = EonaEngine
