# Hotel Review Sentiment Classification Analysis

## Intro
- Objectives: By analyzing reviews on hotel reservation sites, we want to predict the sentiment of new reviews in advance.
- Data: Reviews of “hotels in Seoul” on TripAdvisor, total of 3,126 collected

## Process
- Positive count: 1845 / Negative count: 1293
- Normalization -> Remove all characters except Korean letters and spaces
- Utilize Okt object -> Separate into morphemes and words 
- Remove stop words
  - Use the Korean stop word dictionary provided by RANKS NL
  - Additional and removal of stop words specific to the collected hotel review dataset
- Tokenization: Converting Character Indexes to Vectors
- Remove rare words with word frequency less than 2 times
- Set the maximum length to be used for padding to ‘200’

## Result
1) CNN -> The model shows good performance with a loss error of 0.2594 and an accuracy of 0.8977.
2) LSTM -> The model shows good performance with a loss error of 0.2383 and an accuracy of 0.9108.
