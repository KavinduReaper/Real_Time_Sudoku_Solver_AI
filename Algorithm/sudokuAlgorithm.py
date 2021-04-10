"""
#######################################
    @ Author : The DemonWolf
#######################################
"""

from copy import deepcopy
from itertools import combinations
from typing import List, Tuple, Set


def getBoxSize(size):
    return int(size ** 0.5)


class SudokuClass:
    def __init__(self, grid: List):
        n = len(grid)
        self.grid = grid
        self.n = n
        # Create a Grid of viable candidates for each position
        candidates = []
        for i in range(n):
            Options = []
            for j in range(n):
                if grid[i][j] == 0:
                    Options.append(self.findOptions(i, j, len(grid)))
                else:
                    Options.append(set())
            candidates.append(Options)
        self.candidates = candidates

    def __repr__(self) -> str:
        Represent = ''
        for row in self.grid:
            Represent += str(row) + '\n'
        return Represent

    def getBoxIndices(self, r: int, c: int) -> List[Tuple[int, int]]:
        indices_box = []
        Box_Size = getBoxSize(self.n)
        row_zero = (r // Box_Size) * Box_Size  # get first row index
        col_zero = (c // Box_Size) * Box_Size  # get first column index
        for i in range(row_zero, row_zero + Box_Size):
            for j in range(col_zero, col_zero + Box_Size):
                indices_box.append((i, j))
        return indices_box

    def getBox(self, r: int, c: int) -> List[int]:
        boxValues = []
        for i, j in self.getBoxIndices(r, c):
            boxValues.append(self.grid[i][j])
        return boxValues

    def getRow(self, r: int) -> List[int]:
        return self.grid[r]

    def getColumn(self, c: int) -> List[int]:
        return [row[c] for row in self.grid]

    def findOptions(self, r: int, c: int, Size) -> Set:
        nums = set(range(1, Size + 1))
        setBox = set(self.getBox(r, c))
        setRow = set(self.getRow(r))
        setColumn = set(self.getColumn(c))
        usedNumbers = setRow | setColumn | setBox
        # Return the valid numbers that can be used
        return nums.difference(usedNumbers)

    def counting(self, arr: List[int]) -> List[int]:
        """ Count occurrences in an array... """
        c = [0] * (self.n + 1)
        for x in arr:
            c[x] += 1
        return c

    def allValues(self, arr):
        count = self.counting(arr)
        for num, c in enumerate(count[1:]):  # Exclude 0:
            if c == 0:
                return False, num + 1  # No value or candidate exists
        return True, None

    def allNumbersAreUsed(self, arr):
        """ Make sure that all of the numbers are used, and that they are only used once.... """
        Count = self.counting(arr)
        # Ignore 0
        for c in Count[1:]:
            if c > 1:
                return False  # No value or candidate exists
        return True

    def getCandidates(self, start, end):
        """ Get candidates within two corners of a rectangle/column/row..."""
        candi = set()
        for i in range(start[0], end[0] + 1):
            for j in range(start[1], end[1] + 1):
                candi = candi | self.candidates[i][j]
        return candi

    def puzzleCheckPossibility(self):
        """ Check to see if each row, column, or box contains only unique elements..."""
        rowsSet, colsSet = [], []
        for i in range(self.n):
            # Get rows
            indices_row = [(i, j) for j in range(self.n)]
            rowsSet.append(indices_row)
            # Get columns
            indices_col = [(j, i) for j in range(self.n)]
            colsSet.append(indices_col)
        # Check rows and columns for possibility
        Type = ['row', 'column']
        for t, indices_set in enumerate([rowsSet, colsSet]):
            for k, indices in enumerate(indices_set):
                arr = [self.grid[i][j] for i, j in indices]
                if not self.allNumbersAreUsed(arr):
                    return False, 'Duplicate values found in %s %d' % (Type[t], k)
                arr += list(self.getCandidates(indices[0], indices[-1]))
                possible, missing = self.allValues(arr)
                if not possible:
                    return False, '%d is not placeable in %s %d' % (missing, Type[t], k)
        # Check boxes
        Box_Size = getBoxSize(self.n)
        for i_zero in range(0, self.n, Box_Size):
            for j_zero in range(0, self.n, Box_Size):
                arr = self.getBox(i_zero, j_zero)[:]
                if not self.allNumbersAreUsed(arr):
                    return False, 'Duplicate values in box (%d, %d)' % (i_zero, j_zero)
                for i in range(i_zero, i_zero + Box_Size):
                    for j in range(j_zero, j_zero + Box_Size):
                        arr += list(self.candidates[i][j])
                possible, missing = self.allValues(arr)
                if not possible:
                    return False, '%d not placeable in box (%d, %d)' % (missing, i_zero, j_zero)
        return True, None

    def placeAndErase(self, r: int, c: int, x: int, constraint_prop=True):
        """ In this row, column, and box, remove x as a candidate in the Grid..."""
        # Place candidate x
        self.grid[r][c] = x
        self.candidates[r][c] = set()
        # Candidate x removed from the list of neighbors.
        indices_row = [(r, j) for j in range(self.n)]
        indices_col = [(i, c) for i in range(self.n)]
        indices_box = self.getBoxIndices(r, c)
        # Set of indices for constraint proportion
        erased = [(r, c)]
        erased += self.erase([x], indices_row + indices_col + indices_box, [])
        # Constraint propagation, through every index that was changed
        while erased and constraint_prop:
            i, j = erased.pop()
            indices_row = [(i, j) for j in range(self.n)]
            indices_col = [(i, j) for i in range(self.n)]
            indices_box = self.getBoxIndices(i, j)
            for indices in [indices_row, indices_col, indices_box]:
                uniques = self.getUnique(indices, Type=[1, 2, 3])
                for indices_combo, combo in uniques:
                    # Returning the erased here does not seem to be very helpful.
                    self.setCandidates(combo, indices_combo)
                    erased += self.erase(combo, indices, indices_combo)
            pointers = self.pointingCombos(indices_box)
            for line, indices_pointer, num in pointers:
                erased += self.erase(num, line, indices_pointer)

    def erase(self, nums, indices, keep):
        """ In indices, delete nums as candidates, but not in keep."""
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
        """Set candidates at indices and remove all other candidates"""
        erased = []
        for i, j in indices:
            # Beware triples where the whole triple is not in each box
            old = self.candidates[i][j].intersection(nums)
            if self.candidates[i][j] != old:
                self.candidates[i][j] = old.copy()
                erased.append((i, j))
        return erased

    def countCandidates(self, indices):
        cnt = [[] for _ in range(self.n + 1)]
        # Get counts
        for i, j in indices:
            for num in self.candidates[i][j]:
                cnt[num].append((i, j))
        return cnt

    def getUnique(self, indices, Type=(0, 1, 2)):
        Grp = self.countCandidates(indices)
        uniques = []
        uniques_temp = {2: [], 3: []}  # Potential unique candidates
        for num, groupIndices in enumerate(Grp):
            temp = len(groupIndices)
            if temp == 1 and (1 in Type):
                uniques.append((groupIndices, [num]))
            if temp == 2 and ((2 in Type) or (3 in Type)):
                uniques_temp[2].append(num)
            if temp == 3 and (3 in Type):
                uniques_temp[3].append(num)
        uniques_temp[3] += uniques_temp[2]
        # Check for matching combos (both hidden and naked)
        for temp in [2, 3]:
            if temp not in Type:
                continue
            # Make every possible combination
            for combo in list(combinations(uniques_temp[temp], temp)):
                groupIndices = set(Grp[combo[0]])
                for k in range(1, temp):
                    groupIndices = groupIndices | set(
                        Grp[combo[k]])  # The length will not change if positions are shared.
                if len(groupIndices) == temp:
                    # Unique combo (pair or triple) found
                    uniques.append((list(groupIndices), combo))
        return uniques

    def pointingCombos(self, boxIndices):
        # Indices_box should come from self.getBoxIndices()
        Groups = self.countCandidates(boxIndices)
        Pointers = []
        for num, indices in enumerate(Groups):
            if len(indices) == 2 or len(indices) == 3:
                rowSame, columnSame = True, True
                i_zero, j_zero = indices[0]
                for i, j in indices[1:]:
                    rowSame = rowSame and (i == i_zero)
                    columnSame = columnSame and (j == j_zero)
                if rowSame:
                    line = [(i_zero, j) for j in range(self.n)]
                    Pointers.append((line, indices, [num]))
                if columnSame:
                    line = [(i, j_zero) for i in range(self.n)]
                    Pointers.append((line, indices, [num]))
        return Pointers

    def getAllUnits(self):
        # Get indices for each set
        indicesSet = []
        for i in range(self.n):
            # Row check
            rowIndices = [(i, j) for j in range(self.n)]
            indicesSet.append(rowIndices)
            # Column check
            columnIndices = [(j, i) for j in range(self.n)]
            indicesSet.append(columnIndices)
        return indicesSet

    def getAllBoxes(self):
        boxIndices = []
        for i_zero in range(0, self.n, getBoxSize(self.n)):
            for j_zero in range(0, self.n, getBoxSize(self.n)):
                indices = self.getBoxIndices(i_zero, j_zero)
                boxIndices.append(indices)
        return boxIndices

    def flushCandidates(self) -> None:
        """ According to logical strategies, set candidates across the whole Grid..."""
        boxIndices = self.getAllBoxes()
        indicesSet = self.getAllUnits()
        indicesSet.extend(boxIndices)
        # Repeat this process in case changes are made
        for _ in range(1):
            # Apply strategies
            for indices in indicesSet:
                uniques = self.getUnique(indices, Type=[1, 2])
                for indices_combo, combo in uniques:
                    self.erase(combo, indices, indices_combo)
                    self.setCandidates(combo, indices_combo)
            for indices in boxIndices:
                # Pointing pairs
                pointers = self.pointingCombos(indices)
                for line, indices_pointer, num in pointers:
                    self.erase(num, line, indices_pointer)


def solveSudoku(grid, all_solutions=False):
    def solve(Game, depth=0, progressFactor=1):
        Solved = False
        while not Solved:
            Solved = True
            edited = False
            for i in range(Game.n):
                for j in range(Game.n):
                    if Game.grid[i][j] == 0:
                        Solved = False
                        options = Game.candidates[i][j]
                        if len(options) == 0:
                            return Game.grid, False  # This call is going nowhere
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
                    options = Game.candidates[i][j]
                    # Backtracking check point:
                    progressFactor *= (1 / len(options))
                    for y in options:
                        game_next = deepcopy(Game)
                        game_next.placeAndErase(i, j, y)
                        grid_Final, Solved = solve(game_next, depth=depth + 1, progressFactor=progressFactor)
                        if Solved and not all_solutions:
                            break  # return 1 solution
                    return grid_Final, Solved
        return Game.grid, Solved

    game = SudokuClass(grid)
    # Check for obvious candidates
    game.flushCandidates()

    possible, message = game.puzzleCheckPossibility()
    if not possible:
        print('Error on board. %s' % message)
        return grid, False, message

    grid_final, solved = solve(game, depth=0)
    return grid_final, solved, message
