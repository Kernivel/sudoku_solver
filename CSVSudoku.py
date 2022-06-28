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
    print("Start solving")
    for i in range(81):
        
        #positions
        col = i%9
        row = i//9

        print("Pos :[",str(col),str(row),"]")
        
        #Test solution only if the current case is empty
        if(grid[row][col] == 0):
            #Test the 3 conditions to put a number (Do so for the 9 possible digits)
            #Is my number already in the column
            #Is my nummber already in the row
            #Is my number already in the square (3*3)
            for val in range(1,10):
                #Check if number is in the row
                if val in grid[row]:
                    continue
                #If number was not in the row check if it is in the col
                if val in (grid[0][col],grid[1][col],grid[2][col],grid[3][col],grid[4][col],grid[5][col],grid[6][col],grid[7][col],grid[8][col]):
                    continue
                #If it's not in the col check if it is in the current square
                #Get the current square number
                squareRow = row//3
                squareCol = col//3

                for k in range(3):
                    for l in range(3):
                        kRow = 9*squareRow+k
                        lCol = 9*squareCol+l
                        #If we are checking the same case
                        if kRow == row and lCol == col:
                            continue
                        #Check if the number is already in the case
                        if grid[kRow][lCol] == val:
                            break
                #If the number isn't in the square we have checked all the condition to add this number to the grid.py
                grid[col][row] = val
                #If we are done with the grid : stop there / else do the solving function on the rest of the grid
                if i == 80:
                    return True
                else:
                    if gridSolver(grid):
                        return True
        #No tested digits worked : we have to backtrack
        grid[col][row] = 0

with open('assets/sudoku.csv', newline='') as f:
    reader = csv.reader(f)
    dataCSV = list(reader)
    #print(dataCSV)
    

grid = dataCSV
print("Hello")