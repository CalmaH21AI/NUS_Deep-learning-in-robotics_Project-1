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

# NUS_Deep-learning-in-robotics_Project-2

# Introduction

- The programming environment is Jupyter Notebook.
- The code trains a deep reinforcement learning model using DDPG algorithm.
- The model is trained to reach the non-zero points in a 2D plane with continuous actions.
- The model is expected to find the local nearest point before finding the others.

# Main files

- Complete code: Project2Code.ipynb
- Training code: Project2Code-Training.ipynb
- Testing code:  Project2Code-Testing.ipynb
- Trained model: The four files in "tmp/ddpg"
- Requirements:   requirements.txt

# Main Python libraries

pytorch==1.10.0

torch==1.10.0

torchaudio==0.10.0

torchvision==0.11.1

matplotlib==3.3.4

numpy==1.20.1

# Training & Validation codes

- Code cells in the Complete code:
    - A: main code
    - B: helper functions
    - C: initialisation
    - D: load model
    - E: training
    - F: five plots
    - G: testing
    - H: save model
- Running steps for training in "Project2Code-Training.ipynb": A B C E F H 
- Running steps for testing in "Project2Code-Testing.ipynb":   A B C D G 
