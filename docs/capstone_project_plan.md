High Level (in work):
* Connect Access Database with PostGres so I can do SQL with it (if no good progress after this weekend, consider switching to csv file approach as backup)
* Explore data w/SQL queries to determine where to start for dataset format (i.e. what all columns to include, anything I can clearly remove, what is my target label - classification of a given participant for participate/not participate, or regression for race volume / total participation?  Not sure I have enough data to do volume...each observation would be a race, whereas each observation for the classification problem would be a person attending a given race - lots more data that way)
* Pull data from Access as appropriate to build baseline dataset for use in a Pandas DF
* Exploratory Data analysis
  * Can I answer any of MerGeo's questions at this phase?  Upfront, and/or table for coming back to if time allows after getting actual model put together (primary goal of project)
  * Specifically: geographic trends, demographic trends, etc; see proposal and Eric's Q's
* Data preparation
  * cleaning
  * feature engineering (cross-race type features, distance of participant from race feature?)
  * etc...
* Modeling
  * Try several techniques (logistic regression, RF, gradient boosting)
  * Make sure to do train/validate/test split, cross-validation, regularization as appropriate
* Evaluation
  * Identify specifically how to evaluate models before diving in (loss functions, variation across cross-vals, confusion matrices, ROC curve, cost/benefit matrices, profit curve?)
  * Any opportunity for hypothesis testing on particular aspect(s)?
  * Execute identified evaluation techniques on all models generated to pick best one
  * Are business objectives being met?  Have I answered MerGeo's questions - in model and/or in EDA?
* Deployment
  * Clean, complete, informative GitHub readme covering project and results
  * Convert GitHub readme to a PDF report for MerGeo as well
  * Any reason to deploy to a web app?  Anything interactive for MerGeo?  Anything to back-populate into Dan's Access Database?


CRISP-DM Reminders from Wikipedia:
Business Understanding
This initial phase focuses on understanding the project objectives and requirements from a business perspective, and then converting this knowledge into a data mining problem definition, and a preliminary plan designed to achieve the objectives. A decision model, especially one built using the Decision Model and Notation standard can be used.
Data Understanding
The data understanding phase starts with an initial data collection and proceeds with activities in order to get familiar with the data, to identify data quality problems, to discover first insights into the data, or to detect interesting subsets to form hypotheses for hidden information.
Data Preparation
The data preparation phase covers all activities to construct the final dataset (data that will be fed into the modeling tool(s)) from the initial raw data. Data preparation tasks are likely to be performed multiple times, and not in any prescribed order. Tasks include table, record, and attribute selection as well as transformation and cleaning of data for modeling tools.
Modeling
In this phase, various modeling techniques are selected and applied, and their parameters are calibrated to optimal values. Typically, there are several techniques for the same data mining problem type. Some techniques have specific requirements on the form of data. Therefore, stepping back to the data preparation phase is often needed.
Evaluation
At this stage in the project you have built a model (or models) that appears to have high quality, from a data analysis perspective. Before proceeding to final deployment of the model, it is important to more thoroughly evaluate the model, and review the steps executed to construct the model, to be certain it properly achieves the business objectives. A key objective is to determine if there is some important business issue that has not been sufficiently considered. At the end of this phase, a decision on the use of the data mining results should be reached.
Deployment
Creation of the model is generally not the end of the project. Even if the purpose of the model is to increase knowledge of the data, the knowledge gained will need to be organized and presented in a way that is useful to the customer. Depending on the requirements, the deployment phase can be as simple as generating a report or as complex as implementing a repeatable data scoring (e.g. segment allocation) or data mining process. In many cases it will be the customer, not the data analyst, who will carry out the deployment steps. Even if the analyst deploys the model it is important for the customer to understand up front the actions which will need to be carried out in order to actually make use of the created models.
