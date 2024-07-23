
# * The Scientific Future of Tennis

# The Company
# Azsox Group, S.A. de C.V. is a Mexican-based company dedicated to the study of sports
# with the purpose of improving athletes’ performance; Azsox is also engaged in the technical
# study of the factors that could help to improve the results of athletes in different
# competitions.
# The company has been studying tennis players for several years and gathered a set of variables
# that they consider influence the performance of the tennis player and the number of possible
# titles that they can achieve.
# Tennis as a Competitive Sport
# Carlos Reyes, Azsox’s CEO, was reading some tennis statistics and facts about tennis players
# mentioned in Internet. Some of Carlos’ findings were:
# “Since 1999, professional tennis suffered a major change compared to previous decades.
# Throughout the Open Era of tennis (TennisCompanion, 2023), there have been only a few
# examples of players that became especially dominant, with the German Steffi Graf being
# the only one to achieve more than 20 Major singles titles (Steffi Graf, n.d.). She also won a
# gold medal in the 1988 Summer Olympics. The record for men is 14 Majors, achieved by
# Pete Sampras. The only other player, male or female, besides Graf, to win the Career Golden
# Slam is Andre Agassi, who also won eight Major titles” (Olympedia – Olympians who won
# a Golden Slam in tennis, n.d.).
# “In the 1999 US Open championship, Serena Williams won the first of her 23 Grand Slam
# singles titles, the most amongst women. In that same year, she won the first of her 14 doubles
# titles, all of them alongside her sister, Venus. Both won Olympic gold medals in the “singles”
# and “doubles” categories, and they later won seven Grand Slam “singles” titles” (Olympedia
# – Olympians who won a Golden Slam in tennis, s. f.).
# “Four years after the first victory of Serena Williams, Roger Federer won the first of his 20
# Grand Slam titles in the Wimbledon tournament, which he won seven more times. For many
# years, he held the record of most majors won by a man and still holds several records,
# including the highest number of consecutive weeks ranked as number one in the world in
# tennis. Two years later, Rafael Nadal won his first major’s at the Roland Garros tournament.
# Commonly regarded as the best player in clay surface, he has won 22 Grand Slam titles, 14 of
# which he obtained in the French Open tournament. Nadal has also won two Olympic gold
# medals, both in singles and doubles; he is the second one overall in major titles, only behind
# Novak Djokovic, a 23-time Grand Slam winner, ten-time winner of the Australian Open
# championship and the player with the most non-consecutive weeks ranked as number one in
# the world” (Olympedia – Olympians who won a Golden Slam in tennis, s. f.).
# After reading those news and facts, Carlos met with the complete staff of Azsox’s R&I
# Department. In that meeting, Carlos asked them the following questions: What happened to
# tennis? How is it that Novak Djokovic still dominates the sport in 2024, competing against
# players that are decades younger than he is? Why do the so-called Big Three (Richter, 2023)
# have won 65 of the last 81 singles Grand Slam tournaments? What makes a sports champion?
# More importantly, what makes a tennis world champion? Will we ever see another Serena
# Williams or another person at the same level as hers?

# After those questions, Carlos told them: “I know that we have the data to find the answers to my
# questions. The data we have would not only answer my questions but would also help to prepare a
# presentation for the Mexican Tennis Federation; we have signed a contract with this organization,
# and we need to present them the findings next week. I need you to find how the variables we have
# could provide informed answers; I need the results of the analysis next week, when we will present
# a model to the Mexican Tennis Federation.”
# José Guzmán, leader of Azsox ’s R&I Department, presented to his colleagues some data
# that,according to him,could be useful to accomplish Carlos Reyes’ expectations. Data had
# previously been collected and identified by several persons at Azsox :

# The Independent Variables
# Azsox has identified the variables that could affect the number of titles that a tennis
# player can achieve (see Table 1). 

# Table 1. Variables that could affect the number of titles that a tennis player can achieve.
# Name of the variable Meaning
# AGE Age at which the player became professional
# HEIGHT Player’s height (cm)
# YEARS Number of yearsin the circuit
# WEIGHT Player’s weight (kg)
# WHR Player’s height ratio
# DEX Right (1) or Left (0) handedness
# FSER Average percentage of first serves in (%)
# SPSER Average speed of first serve (mph)
# UERR Average number of unforced errors per set
# EARN Career prize earnings in dollars
# SPONS 2023 Sponsorship earnings in dollars

# The independent variable (CHAMP) is a number calculated as the sum of the following data
# (Source: Players, s. f.):
# 1. Number of Majors/Grand Slam singles titles
# 2. Number of Olympic gold medals
# 3. Number of Olympic silver medals, multiplied by a factor of 0.75
# 4. Number of Masters1000 or WTA1000 titles, multiplied by a factor of 0.75
# 5. Number of ATP500 or WTA500 titles, multiplied by a factor of 0.5
# 6. Number of ATP250 or WTA250 titles, multiplied by a factor of 0.3

# Notes:
# Note #1.-The factors applied to each of the titles are arbitrary, decided by Azsox ’s, based on
# the importance of each of the tournaments (e.g., Grand Slams are the most prestigious and
# difficult tournaments to win).

# Note #2.-The occurrence (e.g., there is a significant greater number of ATP250 and WTA250
# tournaments than Masters 1000 and WTA1000 tournaments) and special consideration for Olympic
# games (they grant no points for players’ rankings but are very prestigious and are held only once
# every four years, hence given the same weight as Majors).

# This database includes information for active ATP players in either the Top 50 ranking as of
# January 2024 and/or winners of, at least, one Grand Slam title (Rankings | Pepperstone ATP
# Rankings (Individual) | ATP Tour | Tennis | ATP Tour | Tennis, s. f.). 

# The Model
# José Guzmán also proposed to the rest of Azsox ’s R&I Department to present a linear model to
# the Mexican Tennis Federation, including the significant variables that most influence athletes’
# performance. According to José, it would have to comply with the Ordinary Least Squares
# efficient model assumptions, most importantly the homoscedasticity of the residuals and the linear
# independence between the regressors, also known as “assumption of no multicollinearity”.

# The final model, according to Jose´s proposal, must be economically significant, as well as tested
# for normality of its residuals and for the global and individual statistical significance of the
# variables’ coefficients.

# Testing and Interpreting the Model
# “The final report that Azsox would deliver, José mentioned, must be tested using a scenario
# analysis where the pessimistic scenario will forecast the possible championships of a player with
# all the variables in percentile 15, the most likely scenario will use a percentile 30 and the optimistic
# scenario will use a percentile 40”. 

# José also proposed to the rest of the team to include in the final report the interpretation of each
# significant coefficient for the Mexican Tennis Federation to make plans on which of the players’
# attributes should be promoted, worked, and improved to have an ATP champion.

# All the members of Azsox’s R&I Department accepted Jose’s proposals and started to work. 

# *** Import required libraries -----------------------------------------------

import pandas as pd # Data manipulation
import numpy as np # Numerical operations
import statsmodels.api as sm # Statistical modeling
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf 
import scipy.stats # Statistical analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor # VIF
from matplotlib import pyplot as plt # Plotting
import datapro # Custom module for model data processing
#       bp_test(res) - Returns a data frame with the Breusch-Pagan test
#       feasible_gls(data,res) - Feasible Generalized Least Squares
#       plot_fit(res,x,y,reg_line=True) - Plot of a OLS regression
#       robust_se (res) - Returns a df with the coeficients

# *** Import data -------------------------------------------------------------

DATA = pd.read_csv("data/SP_TheScientificFutureofTennis_alumn_VF.csv")