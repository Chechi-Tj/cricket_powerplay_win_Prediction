# Cricket Powerplay Win Prediction

This project analyzes how the number of runs scored in the Powerplay (first 6 overs) affects the probability of winning in T20 cricket. Using ball-by-ball data from [Cricsheet](https://cricsheet.org/), we calculate Powerplay totals and determine win percentages across different scoring ranges.

## Project Overview

- **Dataset:** T20 matches in YAML format from Cricsheet  
- **Analysis:** 
  - Calculate Powerplay runs for each team  
  - Label matches as Win/Loss  
  - Bucket Powerplay scores and compute win percentages  
  - Visualize win probability vs Powerplay score  

- **Tools Used:** Python, pandas, matplotlib, PyYAML  

## How to Run

1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Run the analysis:

python src/powerplay_analysis.py  

Check output/figures/ for visualizations and output/tables/ for aggregated results.

Dashbord analysis in progress 
## Future Work
- Add logistic regression / ML model to predict match outcome in real time  
- Compare across IPL teams or T20 international tournaments
- Include wickets lost in Powerplay for a more accurate model

## License
MIT License
