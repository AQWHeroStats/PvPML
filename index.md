## Match Result Prediction Using Logistic Regression Binary Classifier
- We can predict the outcome of a match at each time point in the match by inputting the current score of team A and team B.
- This is done using a machine learning model trained on 3v3 games.
- The model accuracy on training data is 

## Details
- Here, we have parsed data from all completed 3v3 matches on the [Hero PvP System](https://hero.pics/PvP) (currently 40 of them) and generated feature tuples `x1, x2` which represent the match progress and team lead as well as a target vector `y` which encodes a `0` for match loss and `1` for match won. 
- We train on a logistic regression classifier (sklearn) which assigns an associated class probability with each prediction. We report this probability as each team's win chance. 


### Next Steps
- Train a LTSM model on match time to incorporate time series data into model; account for shifts in momentum / "snowball effect" from controlling the middle of the map.
- Train CNN on 20 parameter list (including damage dealt, healing done) along with time series data to see if deeper predictions can be made.
- Perform principle component analysis (PCA) on winning teams to determine which factors most lead to thier victory. 
