
## Match Result Prediction Using Logistic Regression Binary Classifier
- We can predict the outcome of a match at each time point in the match by inputting the current score of team A and team B.
<img width="376" alt="Screen Shot 2021-08-08 at 9 43 19 PM" src="https://user-images.githubusercontent.com/88293729/128661217-77777640-f68e-4749-b42f-1255272ecc28.png">
- This is done using a machine learning model trained on 3v3 games.
- The model accuracy on training data is **78.91%**

## Details

- Here, we have parsed data from all completed 3v3 matches on the [Hero PvP System](https://hero.pics/PvP) (currently 40 of them) and generated feature tuples `x1, x2` which represent the match progress and team lead as well as a target vector `y` which encodes a `0` for match loss and `1` for match won. 
- We train on a logistic regression classifier (sklearn) which assigns an associated class probability with each prediction. We report this probability as each team's win chance. 
- The form of the logistic regression is:
<img width="757" alt="Screen Shot 2021-08-08 at 9 52 44 PM" src="https://user-images.githubusercontent.com/88293729/128661627-f062a17f-f039-4a00-bfd3-9201c2183f7d.png">
- Our learned parameter values are `b1 = -1.10460269`, `b2 = 0.43577768`, `b3 = 0.02632877'.
- To report our predictions:
```python
def predict(TeamAScore, TeamBScore):
    import numpy as np
    predicted_winner = ""
    lead = TeamAScore-TeamBScore
    prog = max(TeamAScore,TeamBScore)/30
    p = 1/(1+np.exp((-1.10460269*prog + 0.43577768*lead + 0.02632877)))
    teamBWin = round(p,4)
    teamAWin = round(1 - teamBWin,4)
    if teamAWin > teamBWin:
        predicted_winner = 'A'
    else:
        predicted_winner = 'B'
    print("Team A Win Probability: "+str(teamAWin*100)+"%")
    print("Team B Win Probability: "+str(teamBWin*100)+"%")
    print("Predicted Winner: "+predicted_winner)
```

## Decision Region Plot
![clf](https://user-images.githubusercontent.com/88293729/128660322-6e5af181-d71f-4db6-8603-c1004874f312.png)
For the most part, the data appears to be linearly seperable with [some outlier games](https://hero.pics/PvP/7618766), and hence a simple linear regression model appears to work decently. The model's prediction improve towards the end of the match compared to the start.

### Next Steps
- Train a LTSM model on match time to incorporate time series data into model; account for shifts in momentum / "snowball effect" from controlling the middle of the map.
- Train CNN on 20 parameter list (including damage dealt, healing done) along with time series data to see if deeper predictions can be made.
- Perform principle component analysis (PCA) on winning teams to determine which factors most lead to thier victory. 
