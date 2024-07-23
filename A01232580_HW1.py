import pandas as pd # Data manipulation
import numpy as np # Numerical operations
import statsmodels.api as sm # Statistical modeling
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf 
import scipy.stats # Statistical analysis
from statsmodels.stats.outliers_influence import variance_inflation_factor # VIF
from matplotlib import pyplot as plt # Plotting
import datapro # Custom module with the following functions:
# bp_test(res) - Returns a data frame with the Breusch-Pagan test
# feasible_gls(data,res) - Feasible Generalized Least Squares
# plot_fit(res,x,y,reg_line=True) - Plot of a OLS regression
# robust_se (res) - Returns a df with the coeficients

# Import the dataset
educat = pd.read_excel("Education.xlsx")
educat.head()
#    Age  Education  Salary
# 0   31          7    12.3
# 1   45         12    45.0
# 2   47          6    13.2
# 3   57          5    12.4
# 4   60         14    28.1

educat.columns
# Index(['Age', 'Education', 'Salary'], dtype='object')

# Define the dependent and independent variables
y_ed = pd.DataFrame(educat.iloc[:,2]) # Dependent variable
X_ed = pd.DataFrame(educat.iloc[:,0:2]) # Independent variables
X_ed = sm.add_constant(X_ed,prepend=True) # Then we add a constant
X_ed.head()
#    const  Age  Education
# 0    1.0   31          7
# 1    1.0   45         12
# 2    1.0   47          6
# 3    1.0   57          5
# 4    1.0   60         14

# Fitting the model
mod1 = sm.OLS( y_ed,X_ed) # OLS model (Ordinary Least Squares)
res1 = mod1.fit()
print(res1.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                 Salary   R-squared:                       0.668
# Model:                            OLS   Adj. R-squared:                  0.661
# Method:                 Least Squares   F-statistic:                     97.57
# Date:                Fri, 12 Jul 2024   Prob (F-statistic):           5.99e-24
# Time:                        13:11:06   Log-Likelihood:                -405.14
# No. Observations:                 100   AIC:                             816.3
# Df Residuals:                      97   BIC:                             824.1
# Df Model:                           2
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const        -12.5828      5.701     -2.207      0.030     -23.897      -1.268
# Age            0.0325      0.109      0.297      0.767      -0.185       0.250
# Education      3.6206      0.266     13.631      0.000       3.093       4.148
# ==============================================================================
# Omnibus:                       18.458   Durbin-Watson:                   1.593
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):               27.976
# Skew:                           0.829   Prob(JB):                     8.42e-07
# Kurtosis:                       4.991   Cond. No.                         197.
# ==============================================================================

# This shows that the variable "Age" is not significant.
# Education has a positive and significant effect on Salary.
# The R-squared is 0.668, which means that 66.8% of the variation in Salary is explained by the model.

# Import the dataset
wage = pd.read_excel("wage1.xlsx")
wage.head()

# Define the dependent and independent variables
y_wg = pd.DataFrame(wage.iloc[:,0]) # Dependent variable
x_wg = pd.DataFrame(wage.drop(columns="wage")) # Independent variables
x_wg = sm.add_constant(x_wg,prepend=True) # Constant
x_wg.head()

# Fitting the model
mod2 = sm.OLS(y_wg,x_wg)
res2 = mod2.fit()
print(res2.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                   wage   R-squared:                       0.890
# Model:                            OLS   Adj. R-squared:                  0.885
# Method:                 Least Squares   F-statistic:                     177.2
# Date:                Fri, 12 Jul 2024   Prob (F-statistic):          5.33e-224
# Time:                        13:20:15   Log-Likelihood:                -851.68
# No. Observations:                 526   AIC:                             1751.
# Df Residuals:                     502   BIC:                             1854.
# Df Model:                          23
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -5.0410      0.422    -11.941      0.000      -5.870      -4.212
# educ           0.0203      0.028      0.731      0.465      -0.034       0.075
# exper         -0.0012      0.018     -0.069      0.945      -0.036       0.034
# tenure         0.0107      0.022      0.484      0.629      -0.033       0.054
# nonwhite      -0.0567      0.185     -0.306      0.760      -0.421       0.307
# female         0.1240      0.134      0.927      0.354      -0.139       0.387
# married       -0.2678      0.132     -2.026      0.043      -0.527      -0.008
# numdep         0.1385      0.049      2.843      0.005       0.043       0.234
# smsa          -0.1984      0.133     -1.491      0.137      -0.460       0.063
# northcen      -0.1844      0.160     -1.152      0.250      -0.499       0.130
# south         -0.1575      0.153     -1.026      0.305      -0.459       0.144
# west           0.0812      0.178      0.456      0.649      -0.269       0.431
# construc      -0.1592      0.296     -0.538      0.591      -0.741       0.422
# ndurman       -0.1044      0.221     -0.473      0.636      -0.538       0.329
# trcommpu      -0.4093      0.307     -1.332      0.184      -1.013       0.195
# trade         -0.0457      0.190     -0.240      0.810      -0.420       0.329
# services       0.2604      0.239      1.089      0.277      -0.210       0.730
# profserv      -0.3108      0.203     -1.530      0.127      -0.710       0.088
# profocc        0.4195      0.170      2.470      0.014       0.086       0.753
# clerocc        0.0844      0.196      0.430      0.667      -0.301       0.470
# servocc        0.6222      0.197      3.154      0.002       0.235       1.010
# lwage          6.5437      0.153     42.724      0.000       6.243       6.845
# expersq     2.987e-05      0.000      0.079      0.937      -0.001       0.001
# tenursq        0.0007      0.001      0.981      0.327      -0.001       0.002
# ==============================================================================
# Omnibus:                      359.649   Durbin-Watson:                   2.136
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4493.297
# Skew:                           2.886   Prob(JB):                         0.00
# Kurtosis:                      16.104   Cond. No.                     7.05e+03
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 7.05e+03. This might indicate that there are
# strong multicollinearity or other numerical problems.

# The variable "expersq" is not significant (p-value = 0.937).
# The R-squared is 0.890, which means that 89% of the variation in wage is explained by the model.

datapro.vif(x_wg, res2, cons=False) # VIF with datapro's function

#     VIF Factor  Variable
# 1     1.978847      educ
# 2    19.703512     exper
# 3     8.639793    tenure
# 4     1.063506  nonwhite
# 5     1.502138    female
# 6     1.399997   married
# 7     1.269618    numdep
# 8     1.194840      smsa
# 9     1.619502  northcen
# 10    1.814374     south
# 11    1.498884      west
# 12    1.283459  construc
# 13    1.652725   ndurman
# 14    1.328219  trcommpu
# 15    2.497060     trade
# 16    1.743128  services
# 17    2.660284  profserv
# 18    2.254774   profocc
# 19    1.806263   clerocc
# 20    1.582314   servocc
# 21    2.225071     lwage
# 22   18.421562   expersq
# 23    7.490425   tenursq

# The variable "expersq" has a high VIF (18.42), which indicates multicollinearity.

# Homework: Test the model but eliminating the variable expersq.
x2_wg = pd.DataFrame(x_wg.drop(columns = "expersq")) # Drop the column "expersq"
x2_wg.head()

# Fitting the model
mod3 = sm.OLS(y_wg,x2_wg)
res3 = mod3.fit()
print(res3.summary())

#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                   wage   R-squared:                       0.890
# Model:                            OLS   Adj. R-squared:                  0.886
# Method:                 Least Squares   F-statistic:                     185.7
# Date:                Fri, 12 Jul 2024   Prob (F-statistic):          3.88e-225
# Time:                        13:25:08   Log-Likelihood:                -851.68
# No. Observations:                 526   AIC:                             1749.
# Df Residuals:                     503   BIC:                             1847.
# Df Model:                          22
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -5.0402      0.422    -11.955      0.000      -5.869      -4.212
# educ           0.0201      0.028      0.728      0.467      -0.034       0.074
# exper          0.0001      0.005      0.021      0.984      -0.010       0.011
# tenure         0.0105      0.022      0.478      0.633      -0.033       0.053
# nonwhite      -0.0567      0.185     -0.306      0.760      -0.420       0.307
# female         0.1230      0.133      0.925      0.356      -0.138       0.384
# married       -0.2699      0.129     -2.089      0.037      -0.524      -0.016
# numdep         0.1375      0.047      2.936      0.003       0.045       0.230
# smsa          -0.1981      0.133     -1.490      0.137      -0.459       0.063
# northcen      -0.1843      0.160     -1.153      0.250      -0.498       0.130
# south         -0.1585      0.153     -1.038      0.300      -0.459       0.142
# west           0.0810      0.178      0.455      0.649      -0.268       0.430
# construc      -0.1581      0.295     -0.535      0.593      -0.739       0.422
# ndurman       -0.1034      0.220     -0.470      0.639      -0.535       0.329
# trcommpu      -0.4075      0.306     -1.331      0.184      -1.009       0.194
# trade         -0.0456      0.190     -0.240      0.811      -0.419       0.328
# services       0.2597      0.239      1.088      0.277      -0.209       0.729
# profserv      -0.3104      0.203     -1.530      0.127      -0.709       0.088
# profocc        0.4201      0.170      2.478      0.014       0.087       0.753
# clerocc        0.0846      0.196      0.431      0.666      -0.301       0.470
# servocc        0.6236      0.196      3.177      0.002       0.238       1.009
# lwage          6.5412      0.149     43.756      0.000       6.247       6.835
# tenursq        0.0007      0.001      1.014      0.311      -0.001       0.002
# ==============================================================================
# Omnibus:                      359.499   Durbin-Watson:                   2.136
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4487.701
# Skew:                           2.884   Prob(JB):                         0.00
# Kurtosis:                      16.095   Cond. No.                     1.93e+03
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 1.93e+03. This might indicate that there are
# strong multicollinearity or other numerical problems.

# The variable "exper" is not significant (p-value = 0.984).
# The R-squared is 0.890, which means that 89% of the variation in wage is explained by the model. It's the same as the previous model.

datapro.vif(x2_wg, res3, cons=False) # VIF with datapro's function

#     VIF Factor  Variable
# 1     1.964412      educ
# 2     1.767703     exper
# 3     8.392302    tenure
# 4     1.063482  nonwhite
# 5     1.487168    female
# 6     1.340496   married
# 7     1.174728    numdep
# 8     1.193991      smsa
# 9     1.619356  northcen
# 10    1.801311     south
# 11    1.498700      west
# 12    1.280792  construc
# 13    1.647104   ndurman
# 14    1.321162  trcommpu
# 15    2.496968     trade
# 16    1.741036  services
# 17    2.657873  profserv
# 18    2.250474   profocc
# 19    1.805939   clerocc
# 20    1.569391   servocc
# 21    2.123878     lwage
# 22    7.233934   tenursq

# The VIF values are similar to the previous model, except for "exper" which is now 1.77 (previously 19.70). This indicates that the multicollinearity problem has been solved.

# Calculate the residuals
res3.resid

# Test for heteroskedasticity
datapro.bp_test(res3) # Custom function

#                                   Breusch-Pagan test
# Lagrange multiplier LM statistic           54.904920
# LM p-value                                  0.000123
# F-value                                     2.664698
# Fp-value                                    0.00007

# This test indicates that there seems to be heteroskedasticity in the model.

# Feasible Generalized Least Squares
datapro.feasible_gls(x2_wg,res3) # Custom function

#                             GLS Regression Results
# ==============================================================================
# Dep. Variable:                      y   R-squared:                       0.904
# Model:                            GLS   Adj. R-squared:                  0.900
# Method:                 Least Squares   F-statistic:                     215.5
# Date:                Fri, 12 Jul 2024   Prob (F-statistic):          1.17e-239
# Time:                        13:45:59   Log-Likelihood:                -655.24
# No. Observations:                 526   AIC:                             1356.
# Df Residuals:                     503   BIC:                             1455.
# Df Model:                          22
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const         -3.1536      0.265    -11.910      0.000      -3.674      -2.633
# educ           0.0037      0.017      0.223      0.823      -0.029       0.037
# exper         -0.0012      0.003     -0.417      0.677      -0.007       0.005
# tenure        -0.0176      0.015     -1.151      0.250      -0.048       0.012
# nonwhite      -0.0226      0.099     -0.227      0.821      -0.218       0.173
# female         0.0154      0.081      0.189      0.850      -0.144       0.175
# married       -0.0634      0.075     -0.842      0.400      -0.212       0.085
# numdep         0.0739      0.029      2.539      0.011       0.017       0.131
# smsa          -0.1050      0.077     -1.357      0.175      -0.257       0.047
# northcen      -0.1092      0.101     -1.079      0.281      -0.308       0.090
# south         -0.0871      0.097     -0.901      0.368      -0.277       0.103
# west           0.1493      0.144      1.037      0.300      -0.134       0.432
# construc      -0.0175      0.174     -0.101      0.920      -0.359       0.324
# ndurman        0.0671      0.138      0.487      0.627      -0.204       0.338
# trcommpu       0.0767      0.158      0.485      0.628      -0.234       0.388
# trade          0.0370      0.124      0.298      0.766      -0.207       0.281
# services       0.5461      0.181      3.011      0.003       0.190       0.902
# profserv      -0.0309      0.132     -0.234      0.815      -0.290       0.229
# profocc        0.2730      0.107      2.543      0.011       0.062       0.484
# clerocc       -0.0738      0.104     -0.711      0.477      -0.278       0.130
# servocc        0.3634      0.129      2.818      0.005       0.110       0.617
# lwage          5.3736      0.101     53.108      0.000       5.175       5.572
# tenursq        0.0016      0.001      2.566      0.011       0.000       0.003
# ==============================================================================
# Omnibus:                      328.323   Durbin-Watson:                   2.092
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):             3122.646
# Skew:                           2.656   Prob(JB):                         0.00
# Kurtosis:                      13.690   Cond. No.                     1.40e+03
# ==============================================================================

# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 1.4e+03. This might indicate that there are
# strong multicollinearity or other numerical problems.

# The R-squared is 0.904, which means that 90.4% of the variation in wage is explained by the model.
# On the other hand, we now 

# Plot the fit
datapro.plot_fit(res3,x2_wg,y_wg,reg_line=True) # Custom function

# Calculate the robust standard errors
datapro.robust_se(res3) # Custom function

#               coef   std err  HC0 std err  HC1 std err
# const    -5.040191  0.421605     0.483294     0.494220
# educ      0.020083  0.027598     0.027603     0.028227
# exper     0.000110  0.005341     0.003677     0.003760
# tenure    0.010454  0.021864     0.024246     0.024794
# nonwhite -0.056659  0.185080     0.132231     0.135221
# female    0.122951  0.132974     0.123032     0.125814
# married  -0.269909  0.129207     0.104280     0.106637
# numdep    0.137501  0.046831     0.039571     0.040465
# smsa     -0.198108  0.132921     0.112316     0.114855
# northcen -0.184280  0.159881     0.162587     0.166263
# south    -0.158523  0.152734     0.142471     0.145692
# west      0.081012  0.177862     0.229882     0.235079
# construc -0.158091  0.295422     0.269499     0.275592
# ndurman  -0.103350  0.219914     0.233634     0.238916
# trcommpu -0.407513  0.306190     0.266624     0.272651
# trade    -0.045603  0.190267     0.230566     0.235778
# services  0.259699  0.238779     0.244355     0.249879
# profserv -0.310360  0.202827     0.257900     0.263730
# profocc   0.420117  0.169550     0.151557     0.154983
# clerocc   0.084617  0.196126     0.149068     0.152438
# servocc   0.623573  0.196264     0.195174     0.199586
# lwage     6.541165  0.149492     0.325213     0.332565
# tenursq   0.000746  0.000735     0.000904     0.000925

# The robust standard errors are larger than the standard errors, which indicates that the standard errors are underestimated. This is due to the presence of heteroskedasticity in the model.