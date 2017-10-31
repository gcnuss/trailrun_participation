## Work During Capstone:
Data preparation, cleaning, transformation to get into a state to support analysis

What are our customers' tastes regarding types of events?
- Clustering or other unsupervised learning if data available to ID key factors
- EDA to look at attendance across types, distances, etc.

What are their patterns of participation in our events?
- Recommender system to recommend top events for each person; use recommendation lists to pull insights on 'best events', groups of people for which events are recommended, etc.
- EDA, clustering analysis to identify participation patterns by demographics (M/F, age ranges, past participation, time of year, location, etc)

## Work After Capstone, Before Mid-Nov Meeting:

Where do they live, and how does geography affect event participation?
- Hypothesis testing for whether geography affects participation
- Assuming it does, could look at geography predictor (e.g. distance from home to event) for predicting participation

What times of day and days of the week do people tend to register for events?
- Classification model to predict registration day (or linear regression to predict day & time?)
- Definite opportunity for EDA to look at what we have in the existing data for peaks / valleys (then will need to do ML to understand what predictors drive those peaks / valleys)

Additional EDA for patterns of participation:
- EDA to explore the high percentage of participants who have only attended one event in the past 2.5 years
- EDA, clustering analysis to identify participation patterns by demographics (M/F, age ranges, past participation, time of year, location, etc)

## Additional Future Work (TBD Timing):
What other events do they participate in?
- Web scraping of participants by name (?) from results of other companies' races
- EDA to add features that identify other Mergeo event types attended and using web scraping also what other event types are participated in
- SQL DB mining to optimize the data we have from Dan and look at joins, etc (?)

What email and social media messages do people respond to and engage with the most strongly?
- A/B hypothesis testing
- Multi armed bandit bayesian statistics; bayesian ML techniques / algorithms (different options may work better for different demographics)

Could also look at churn prediction, define a person as inactive if no events after X amount of time, and then look at what drives people to remain active vs. inactive!
