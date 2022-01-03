# NUS_Deep-learning-in-robotics_Project-1

# Instruction

- Libraries involved:
    - numpy
    - random
    - time -> to record the process time
    - IPython.display -> to clear output and display movement
    - matplotlib.pyplot -> to plot scatter diagram
    - pandas -> to plot scatter diagram
    - tabulate -> to display Q-table
    - seaborn -> to create heatmap
- Functions in "fl" class:
    - show_grid(): display the map
    - show_Q(): display the Q-table
    - reset(): clear all learning outcomes
- There are three methods to run the codes.
- Complete codes and comments are in frozenlake.py

## Method 1
- Files involved
    1. frozenlake.py
    2. main_4x4.py
    3. main_10x10.py
- Description
    - All algorithms are compiled in "frozenlake.py"
    - Running "main_xxx.py" files can obtain the results in one go for all three learning techniques.
    - Parameters can be modified in "main_xxx.py" files.
    - For "main_10x10.py" file:
        - Comment "fails = None" and uncomment "fails = [...]" to create a random map.
        - Uncomment "fails = None" and comment "fails = [...]" to employ the map used in the report.
    
## Method 2
- Files involved
    1. frozenlake.py
    2. frozenlake_import.ipynb
- Description
    - Same as Method 1 but implemented in jupyter IDE. 
    
## Method 3
- Files involved
    1. frozenlake_complete.ipynb
- Description
    - All codes are compiled in one file.
    - Robot's movement in the grid can be shown in the output.
