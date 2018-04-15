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

### A basic check of the pipes

Let's check the basics - try a linear regression for time to pregnancy on just 
the numerical stuff. This will check the X predicting y pipes, and we can use a L1 
penalty to immediately enforce scarcity. Coefficients found:

            Age    NumBMI  TempLogFreq  SexLogFreq  AnovCycles 
           0.0  0.000000    -0.613246   -5.892810     2.070549
           0.0 -0.000057    -1.926542    1.261366    -0.083681

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
      
### Explicit example

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
Ok, now trying a boosted tree approach from https://github.com/dmlc/xgboost

This gets a higher 32% score, but I'm having to dig to find out where the explanatory power 
is coming from.




    