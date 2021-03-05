from copy import deepcopy
from itertools import combinations
from typing import List, Tuple, Set


def getBoxSize(size):
    return int(size ** 0.5)


class Sudoku:
    def __init__(self, grid: List):
        n = len(grid)
        self.grid = grid
        self.n = n
        # Create a Grid of viable candidates for each position
        candidates = []
        for i in range(n):
            row = []
            for j in range(n):
                if grid[i][j] == 0:
                    row.append(self.findOptions(i, j, len(grid)))
                else:
                    row.append(set())
            candidates.append(row)
        self.candidates = candidates

    def __repr__(self) -> str:
        repr = ''
        for row in self.grid:
            repr += str(row) + '\n'
        return repr

    def getRow(self, r: int) -> List[int]:
        return self.grid[r]

    def getCol(self, c: int) -> List[int]:
        return [row[c] for row in self.grid]

    def getBoxIndices(self, r: int, c: int) -> List[Tuple[int, int]]:
        indices_box = []
        Box_Size = getBoxSize(self.n)
        i_zero = (r // Box_Size) * Box_Size  # get first row index
        j_zero = (c // Box_Size) * Box_Size  # get first column index
        for i in range(i_zero, i_zero + Box_Size):
            for j in range(j_zero, j_zero + Box_Size):
                indices_box.append((i, j))
        return indices_box

    def getBox(self, r: int, c: int) -> List[int]:
        box = []
        for i, j in self.getBoxIndices(r, c):
            box.append(self.grid[i][j])
        return box

    def findOptions(self, r: int, c: int, Size) -> Set:
        nums = set(range(1, Size + 1))
        set_row = set(self.getRow(r))
        set_col = set(self.getCol(c))
        set_box = set(self.getBox(r, c))
        used = set_row | set_col | set_box
        valid = nums.difference(used)
        return valid

    def counting(self, arr: List[int]) -> List[int]:
        """ Count occurrences in an array... """
        c = [0] * (self.n + 1)
        for x in arr:
            c[x] += 1
        return c

    def allUnique(self, arr: List[int]) -> bool:
        """ Verify that all numbers are used, and at most once... """
        count = self.counting(arr)
        # ignore 0
        for c in count[1:]:
            if c != 1:
                return False
        return True

    def checkDone(self) -> bool:
        """ Check if each row/column/box only has unique elements"""
        # Check rows
        for i in range(self.n):
            if not self.allUnique(self.getRow(i)):
                return False
            # Check columns
            if not self.allUnique(self.getCol(i)):
                return False
        # Check boxes
        for i_zero in range(0, self.n, getBoxSize(self.n)):
            for j_zero in range(0, self.n, getBoxSize(self.n)):
                if not self.allUnique(self.getBox(i_zero, j_zero)):
                    return False
        return True

    def allValues(self, arr):
        count = self.counting(arr)
        for num, c in enumerate(count[1:]):  # Exclude 0:
            if c == 0:
                return False, num + 1  # No value or candidate exists
        return True, None

    def noDuplicates(self, arr):
        Count = self.counting(arr)
        for c in Count[1:]:  # Exclude 0:
            if c > 1:
                return False  # No value or candidate exists
        return True

    def getCandidates(self, start, end):
        """ Get candidates within two corners of a rectangle/column/row"""
        candi = set()
        for i in range(start[0], end[0] + 1):
            for j in range(start[1], end[1] + 1):
                candi = candi | self.candidates[i][j]
        return candi

    def checkPossible(self):
        """ Check if each row/column/box can have all unique elements"""
        rows_set, cols_set = [], []
        for i in range(self.n):
            # Get rows
            indices_row = [(i, j) for j in range(self.n)]
            rows_set.append(indices_row)
            # Get columns
            indices_col = [(j, i) for j in range(self.n)]
            cols_set.append(indices_col)
        # Check rows and columns
        Type = ['row', 'column']
        for t, indices_set in enumerate([rows_set, cols_set]):
            for k, indices in enumerate(indices_set):
                arr = [self.grid[i][j] for i, j in indices]
                if not self.noDuplicates(arr):
                    return False, 'Duplicate values in %s %d' % (Type[t], k)
                arr += list(self.getCandidates(indices[0], indices[-1]))
                possible, missing = self.allValues(arr)
                if not possible:
                    return False, '%d not placeable in %s %d' % (missing, Type[t], k)
        # Check boxes
        Box_Size = getBoxSize(self.n)
        for i_zero in range(0, self.n, Box_Size):
            for j_zero in range(0, self.n, Box_Size):
                arr = self.getBox(i_zero, j_zero)[:]
                if not self.noDuplicates(arr):
                    return False, 'Duplicate values in box (%d, %d)' % (i_zero, j_zero)
                for i in range(i_zero, i_zero + Box_Size):
                    for j in range(j_zero, j_zero + Box_Size):
                        arr += list(self.candidates[i][j])
                possible, missing = self.allValues(arr)
                if not possible:
                    return False, '%d not placeable in box (%d, %d)' % (missing, i_zero, j_zero)
        return True, None

    # # ------- Candidate functions -------- ##
    def placeAndErase(self, r: int, c: int, x: int, constraint_prop=True):
        """ Remove x as a candidate in the Grid in this row, column and box"""
        # place candidate x
        self.grid[r][c] = x
        self.candidates[r][c] = set()
        # remove candidate  x in neighbours
        indices_row = [(r, j) for j in range(self.n)]
        indices_col = [(i, c) for i in range(self.n)]
        indices_box = self.getBoxIndices(r, c)
        erased = [(r, c)]  # set of indices for constraint proportion
        erased += self.erase([x], indices_row + indices_col + indices_box, [])
        # print(erased)
        # constraint propagation, through every index that was changed
        while erased and constraint_prop:
            i, j = erased.pop()
            indices_row = [(i, j) for j in range(self.n)]
            indices_col = [(i, j) for i in range(self.n)]
            indices_box = self.getBoxIndices(i, j)
            for indices in [indices_row, indices_col, indices_box]:
                uniques = self.getUnique(indices, Type=[1, 2, 3])
                # print(uniques)
                for indices_combo, combo in uniques:
                    # passing back the erased here doesn't seem to be very helpful
                    self.setCandidates(combo, indices_combo)
                    erased += self.erase(combo, indices, indices_combo)
            pointers = self.pointingCombos(indices_box)
            for line, indices_pointer, num in pointers:
                erased += self.erase(num, line, indices_pointer)

    def erase(self, nums, indices, keep):
        """ Erase nums as candidates in indices, but not in keep"""
        erased = []
        for i, j in indices:
            edited = False
            if (i, j) in keep:
                continue
            for x in nums:
                if x in self.candidates[i][j]:
                    self.candidates[i][j].remove(x)
                    edited = True
            if edited:
                erased.append((i, j))
        return erased

    def setCandidates(self, nums, indices):
        """set candidates at indices. Remove all other candidates"""
        erased = []
        for i, j in indices:
            # beware triples where the whole triple is not in each box
            old = self.candidates[i][j].intersection(nums)
            if self.candidates[i][j] != old:
                self.candidates[i][j] = old.copy()
                erased.append((i, j))  # made changes here
        return erased

    def countCandidates(self, indices):
        count = [[] for _ in range(self.n + 1)]
        # get counts
        for i, j in indices:
            for num in self.candidates[i][j]:
                count[num].append((i, j))
        return count

    def getUnique(self, indices, Type=(0, 1, 2)):
        groups = self.countCandidates(indices)
        uniques = []  # final set of unique candidates to return
        uniques_temp = {2: [], 3: []}  # potential unique candidates
        for num, group_indices in enumerate(groups):
            c = len(group_indices)
            if c == 1 and (1 in Type):
                uniques.append((group_indices, [num]))
            if c == 2 and ((2 in Type) or (3 in Type)):
                uniques_temp[2].append(num)
            if c == 3 and (3 in Type):
                uniques_temp[3].append(num)
        uniques_temp[3] += uniques_temp[2]
        # check for matching combos (both hidden and naked)
        for c in [2, 3]:
            if c not in Type:
                continue
            for combo in list(combinations(uniques_temp[c], c)):  # make every possible combination
                group_indices = set(groups[combo[0]])
                for k in range(1, c):
                    group_indices = group_indices | set(
                        groups[combo[k]])  # if positions are shared, this will not change the length
                if len(group_indices) == c:
                    # unique combo (pair or triple) found
                    uniques.append((list(group_indices), combo))
        return uniques

    def pointingCombos(self, indices_box):
        # indices_box should come from self.get_indices_box()
        groups = self.countCandidates(indices_box)
        pointers = []
        for num, indices in enumerate(groups):
            # need a pair or triple
            if len(indices) == 2 or len(indices) == 3:
                row_same, col_same = True, True
                i_zero, j_zero = indices[0]
                for i, j in indices[1:]:
                    row_same = row_same and (i == i_zero)
                    col_same = col_same and (j == j_zero)
                if row_same:
                    line = [(i_zero, j) for j in range(self.n)]
                    pointers.append((line, indices, [num]))
                if col_same:
                    line = [(i, j_zero) for i in range(self.n)]
                    pointers.append((line, indices, [num]))
        return pointers

    def boxLineReduction(self, indices_box):
        keeps = []
        i_zero, j_zero = indices_box[0]
        i_one, j_one = min(i_zero + getBoxSize(self.n), self.n - 1), min(j_zero + getBoxSize(self.n), self.n - 1)
        # check rows
        for i in range(i_zero, i_one + 1):
            row = self.getCandidates((i, j_zero), (i, j_one))
            line = self.getCandidates((i, 0), (i, j_zero - 1)) | self.getCandidates((i, j_one + 1), (i, self.n - 1))
            uniques = row.difference(line)
            if uniques:
                keeps.append(([(i, j) for j in range(j_zero, j_one + 1)], list(uniques)))
        # check columns
        for j in range(j_zero, j_one + 1):
            col = self.getCandidates((i_zero, j), (i_one, j))
            line = self.getCandidates((0, j), (i_zero - 1, j)) | self.getCandidates((i_one + 1, j), (self.n - 1, j))
            uniques = col.difference(line)
            if uniques:
                keeps.append(([(i, j) for i in range(i_zero, i_one + 1)], list(uniques)))
        return keeps

    def getAllUnits(self):
        # get indices for each set
        indices_set = []
        for i in range(self.n):
            # check in rows
            indices_row = [(i, j) for j in range(self.n)]
            indices_set.append(indices_row)
            # check in column
            indices_col = [(j, i) for j in range(self.n)]
            indices_set.append(indices_col)
        return indices_set

    def getAllBoxes(self):
        indices_box = []
        for i_zero in range(0, self.n, getBoxSize(self.n)):
            for j_zero in range(0, self.n, getBoxSize(self.n)):
                indices = self.getBoxIndices(i_zero, j_zero)
                indices_box.append(indices)
        return indices_box

    def flushCandidates(self) -> None:
        """set candidates across the whole Grid, according to logical strategies"""
        # get indices for each set
        indices_box = self.getAllBoxes()
        indices_set = self.getAllUnits()
        indices_set.extend(indices_box)
        # repeat this process in case changes are made
        for _ in range(1):
            # apply strategies
            for indices in indices_set:
                # hidden/naked singles/pairs/triples
                uniques = self.getUnique(indices, Type=[1, 2])
                for indices_combo, combo in uniques:
                    self.erase(combo, indices, indices_combo)
                    self.setCandidates(combo, indices_combo)
            for indices in indices_box:
                # pointing pairs
                pointers = self.pointingCombos(indices)
                for line, indices_pointer, num in pointers:
                    self.erase(num, line, indices_pointer)


def solveSudoku(grid, verbose=True, all_solutions=False):
    def solve(Game, depth=0, progress_factor=1):
        Solved = False
        while not Solved:
            Solved = True  # assume Solved
            edited = False  # if no edits, either done or stuck
            for i in range(Game.n):
                for j in range(Game.n):
                    if Game.grid[i][j] == 0:
                        Solved = False
                        options = Game.candidates[i][j]
                        if len(options) == 0:
                            return Game.grid, False  # this call is going nowhere
                        elif len(options) == 1:  # Step 1
                            Game.placeAndErase(i, j, list(options)[0])  # Step 2
                            edited = True
            if not edited:  # changed nothing in this round -> either done or stuck
                if Solved:
                    return Game.grid, True
                else:
                    # Find the box with the least number of options and take a guess
                    # The erase() changes this dynamically in the previous for loop
                    min_guesses = (Game.n + 1, -1)
                    for i in range(Game.n):
                        for j in range(Game.n):
                            options = Game.candidates[i][j]
                            if min_guesses[0] > len(options) > 1:
                                min_guesses = (len(options), (i, j))
                    i, j = min_guesses[1]
                    # print(min_guesses)
                    options = Game.candidates[i][j]
                    # backtracking check point:
                    progress_factor *= (1 / len(options))
                    for y in options:
                        game_next = deepcopy(Game)
                        game_next.placeAndErase(i, j, y)
                        grid_Final, Solved = solve(game_next, depth=depth + 1, progress_factor=progress_factor)
                        if Solved and not all_solutions:
                            break  # return 1 solution
                    return grid_Final, Solved
        return Game.grid, Solved

    game = Sudoku(grid)
    game.flushCandidates()  # check for obvious candidates

    possible, message = game.checkPossible()
    if not possible:
        print('Error on board. %s' % message)
        return grid, False, message

    grid_final, solved = solve(game, depth=0)
    return grid_final, solved, message
