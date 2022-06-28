##Quick debug file to test the BT Alogrithm
from asyncio import threads
from pickletools import uint8
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PySimpleGUI as sg
import ast
import csv

def gridSolver(grid):
    for i in range(81):
        #positions
        col = i%9
        row = i//9
        squareRow = row//3
        squareCol = col//3
        print("At i :",str(i),"val is :",str(grid[row][col]),"Pos :[",str(row),",",str(col),"] Square :(",str(squareRow),",",str(squareCol),")")
        
        #Test solution only if the current case is empty
        if grid[row][col] == 0:
        #     #Test the 3 conditions to put a number (Do so for the 9 possible digits)
        #     #Is my number already in the column
        #     #Is my nummber already in the row
        #     #Is my number already in the square (3*3)
            for newDigit in range(1,10):
                #print("Testing ",str(newDigit))
                if newDigit not in grid[row]:
                    #print(str(newDigit), "not in row")
                    if newDigit not in (grid[0][col],grid[1][col],grid[2][col],grid[3][col],grid[4][col],grid[5][col],grid[6][col],grid[7][col],grid[8][col]):
                        #print(str(newDigit), "not in col")
                        startRow = squareRow*3
                        startCol = squareCol*3
                        flagValueInSquare = False
                        for k in range(startRow,startRow+3):
                            for l in range(startCol,startCol+3):
                                if newDigit == grid[k][l]:
                                    flagValueInSquare = True
                        if not flagValueInSquare:
                            #print(str(newDigit), "not in case")
                            grid[row][col] = newDigit
                            #print("Changed to :",str(grid[row][col]))
                            if checkGrid(grid):
                                return True
                            if gridSolver(grid):
                                return True
            break
    grid[row][col] = 0
            
            
            
    
#A function to check if the grid is full
def checkGrid(grid):
    for row in range(0,9):
        for col in range(0,9):
            if grid[row][col]==0:
                return False
    return True

#A backtracking/recursive function to check all possible combinations of numbers until a solution is found
def solveGrid(grid):
  #Find next empty cell
  for i in range(0,81):
    row=i//9
    col=i%9
    if grid[row][col]==0:
      for value in range (1,10):
        #Check that this value has not already be used on this row
        if not(value in grid[row]):
          #Check that this value has not already be used on this column
          if not value in (grid[0][col],grid[1][col],grid[2][col],grid[3][col],grid[4][col],grid[5][col],grid[6][col],grid[7][col],grid[8][col]):
            #Identify which of the 9 squares we are working on
            square=[]
            if row<3:
              if col<3:
                square=[grid[i][0:3] for i in range(0,3)]
              elif col<6:
                square=[grid[i][3:6] for i in range(0,3)]
              else:  
                square=[grid[i][6:9] for i in range(0,3)]
            elif row<6:
              if col<3:
                square=[grid[i][0:3] for i in range(3,6)]
              elif col<6:
                square=[grid[i][3:6] for i in range(3,6)]
              else:  
                square=[grid[i][6:9] for i in range(3,6)]
            else:
              if col<3:
                square=[grid[i][0:3] for i in range(6,9)]
              elif col<6:
                square=[grid[i][3:6] for i in range(6,9)]
              else:  
                square=[grid[i][6:9] for i in range(6,9)]
            #Check that this value has not already be used on this 3x3 square
            if not value in (square[0] + square[1] + square[2]):
              grid[row][col]=value
              if checkGrid(grid):
                print("Grid Complete and Checked")
                return True
              else:
                if solveGrid(grid):
                  return True
      break
  print("Backtrack")
  grid[row][col]=0  

with open('assets/sudoku.csv', newline='') as f:
    reader = csv.reader(f)
    dataCSV = list(reader)
    

grid = [list(map(int,i) ) for i in dataCSV]
gridSolver(grid)
print(grid)
