# NaturalCycles
Quick play with the Natural Cycles data Ok, let's have a look at the data... 
numerical looks as expected, as does textual

                Country   Pill     FPlength    CycleVar    ExitStatus
    count       15294  12005        15294       15294      15294
    unique      80      2           3           2          3
    top         SE      False       normal      regular    Dropout
    freq     11864      10189       10165        8212       7651

Note, before we go any further I'm going to set aside 20% of the data until the very end.
This is because I'm going to try to make some predictions, so this test data will give me 
an idea of the out-of-sample power of my models.

#### A basic check of the pipes

Let's check the basics - try a linear regression for time to pregnancy on just 
the numerical stuff. This will check the X predicting y pipes, and we can use a L1 
penalty to immediately enforce scarcity. Coefficients found:

            Age    NumBMI  TempLogFreq  SexLogFreq  AnovCycles 
           0.0  0.000000    -0.613246   -5.892810     2.070549
           0.0 -0.000057    -1.926542    1.261366    -0.083681

validation r2_score is 13.2%

Cool, pipes working so let's have a think: 
* Predicting the length of time to get pregnant, so negative coeffs are better
* I would expect age to be a bigger effect,
* Where has weight gone?
* Wasn't there something about BMI being 0 instead of NaN...?

Ah, need to normalize regressors obvo! And put weight back in. And get rid of 0 BMIs/ weights
(prob helps to read the email)... Nope, I did normalize but the others are better predictors.
As expected being older doesn't help.

        Age    NumBMI    Weight  TempLogFreq  SexLogFreq  AnovCycles 
      0.000000 -0.000000  0.006235    -0.670034    -6.94606     2.244467
      0.000036 -0.000507  0.000000    -1.912383     3.82458    -0.144918
      
#### Explicit example

Still didn't trust the age coefficient so really dug in for an explicit example, but all 
looking good:

    Country               BR
    Age                   35
    NumBMI           25.7812
    Pill               False
    NCbefore           False
    FPlength          normal
    Weight                66
    CycleVar         regular
    TempLogFreq         0.16
    SexLogFreq          0.08
    DaysTrying            24
    CyclesTrying           0
    ExitStatus      Pregnant

The simple L1 penalty gets this totally wrong, predicting 4.4 cycles instead of 0. However, 
it's not all because of the age (sum the entries to get the 4.4)

    Age  NumBMI  Weight  TempLogFreq  SexLogFreq  AnovCycles   Intercept
    0.0    -0.0     0.4         -0.1        -0.6          0.0        2.5
    0.0    -0.3     0.0         -0.0         0.0         -0.0        2.5

## Handling categorical data
Currently no validation of parameters (hence jump from quoted 32% to 36% since last commit for boosted tree)

#### Random Forest
Random forest and categorical data. Important features:

        Category      Feature  Importance    cumsum
    105  TempLogFreq  TempLogFreq    0.201972  0.201972
    107  AnovCycles   AnovCycles     0.143699  0.345670
    106   SexLogFreq   SexLogFreq    0.136289  0.481959
    96       regular     CycleVar    0.084658  0.566617
    87       LongAgo         Pill    0.081986  0.648603
    90         False     NCbefore    0.058337  0.706940
    102          Age          Age    0.039618  0.746557
    95     irregular     CycleVar    0.034207  0.780765
    89        Recent         Pill    0.032954  0.813718
    53            MY      Country    0.032256  0.845974

validation r2_score is 20.7%

#### Boosted tree
And with a boosted tree from https://github.com/dmlc/xgboost

Important features:

        Category      Feature  Importance    cumsum
    106   SexLogFreq   SexLogFreq    0.269572  0.269572
    105  TempLogFreq  TempLogFreq    0.215327  0.484899
    103       NumBMI       NumBMI    0.131931  0.616829
    102          Age          Age    0.101578  0.718407
    104       Weight       Weight    0.082945  0.801352
    107  AnovCycles   AnovCycles     0.078137  0.879489
    89        Recent         Pill    0.024493  0.903982
    94         short     FPlength    0.016529  0.920511
    95     irregular     CycleVar    0.012472  0.932983
    92          long     FPlength    0.010669  0.943651

validation r2_score is 36.0%


Ok, now moving on to drop out.... Hmmm... negative r2 is always
a bad sign, hopefully will have a bit more time tonight.

    