# sudoku_solver-WIP-
Sudoku solver using openCV and TensorFlow (Python)

The goal is to be able to recognize a sudoku grid using the user's webcam, extract its features and then solve it.

1. OpenCV is used to open the camera and detect the grid itself
2. TensorFlow with MNSIT trained model will then try to identify each cell containing a number
3. We need to apply an algorithm to solve the grid itself
4. We send back the solution as an image to the user.
