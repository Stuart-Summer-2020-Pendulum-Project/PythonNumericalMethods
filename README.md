# PythonNumericalMethods

## Purpose

Prompts the user for 2 inputs: # of experiment to analyze, and the type of data to analyze. This python script will then solve the pendulum motion numerically, while also finding the best parameters that will minize error with respect to the selected data. The program will plot this optimized fit against the data and ChiSq/MeanSq error values. Predicted values and standard deviations of g and B (drag coefficient) will be displayed in the terminal.

Program can be used and understood by students that have working knowledge of OS directories and running python through linux/terminal. The approximate run-time of the script is 20 seconds. 

### Python Libraries
- pandas
- numpy
- matplotlib
- os
- scipy.optimize.curve_fit
- datetime
- sklearn.metrics
- scipy.stats


## Contents
- final_analysis.py 
The script that fits an accurate model onto our raw data.

- SummerResearch_Backtest.py
Applies the same fit, but outputs metrics on 4 evenly split partitions of the dataset. Outputs metrics for each partition


## Walk through of code logic
1) Pulls data and cleans it up a little by selecting the 300th entry and beyond
2) Based on user inputs, sets variables that will be used later for fitting specifications
3) Conical Pendulum Function
- Has inputs of x: time, amp: amplitude, g: gravity, B: drag coefficient, and shift: vertical shift
- Solves the pendulum motion numerical using lagragian methods
- Returns an array of generated solution
4) Plotting
- Apply the curve_fit library using the Conical Pendulum Function
- Calculates error between generated predicted values and actual target values from data
- Plot Fit VS Data with error metrics in the legend
- Returns the "best" (optimized) Conical Pendulum inputs in the terminal

### Current Issues
- Program struggles when fitting distance data
- Run time is a little long
- Backtest script validates that our fit does not necessary fall within standard deviations when applied to different segments in the dataset
