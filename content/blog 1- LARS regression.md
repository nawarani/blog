Title: Least Angle Regression (LARS)
Date: 2019-09-14
Slug: blog-1

Before we get into what lars regression is, we must take a peek into forward selection and forward stagewise regression. For simplicity, we will be using a simplistic example of black friday shopping. You are shopping on black friday since there is sale on everything. We are modelling your path as you buy these items: A keyboard, a mouse and a graphics card(in order of priority, graphics card is optional). Keyboard will be bought from best buy and the graphics card and cooling system will be bought from canada computers.

**Forward selection** is a selection method where in each step the model takes in the variable with highest correlation entirely and continues to predict. This is great, except when there are two correlated variables. In the case of multiple correlated variables, forward selection ends up ignoring the other ones, since after adding one of the correlated variable, the rest don't offer much explanation to the model anymore.

Here, you've gone into best buy to buy a keyboard, since it is your biggest concern. Since the mouse is highly likely to be found in best buy as well, forward selection will skip mouse and predict that you will go to canada computers next. Since it will skip the path inside best buy towards the mouse section, it won't be as accurate. 

**Forward stagewise regression** on the other hand, solves this problem by only adding a predetermined amount of a variable. Since it doesn't add the entirity of a variable, other correlated variables are still considered and added in accordingly. But due to the process, forward stagewise ends up being very inefficient when there are a lot of variables to consider.

In this case, our model is being very cautious of your path and updating it every step at a time. Since you are likely to go into best buy to find keyboard first, it will predict your path to best buy one step at a time. At every step it will reevaluate whether you are closer to the store for keyboard, mouse or graphics card, and update its path accordingly. This is great, since this time around the model will be more accurate in its prediction. But if we had 50 things in our list, this model would quickly become burdensome, having to calculate at each step. 

LARS or **Least Angle Regression** aims to solve the limitation of both of the previously mentioned methods. Instead of moving in small predetermined amounts of a variable, it hops towards the most correleated variable until another variable becomes just as correlated. At this stage it changes direction in a way that it forms equal angles(equal correlation) with each variable. Thus the name, least angle regression.

In case of LARS, we will be keeping mouse and graphics card as well, but this time we will modelling our path towards best buy for the keyboard, until we reach best buy. At best buy, buying both mouse and keyboard are equally likely to be bought, so the model will predict a path equally distant from the mouse section and the keyboard. So if you decide to buy the mouse first and then the keyboard, the models path will be a closer approximation than a model that only predicts your path to keyboard. Afterward buying the keyboard and the mouse you're equally likely to go to home or to canada computers. The model will continue to model a equidistant path for your home and canada computers. Since the path revision only happens when two variables becaome equally correlated, it's more accurate than forward selection and lighter/more efficient than forward stagewise regession.

This may sound similar to Lasso, which it is, since Lasso operates in a similar way. The difference in Lasso is that it drops a variable once its prediction capacity hits zero. LARS can be modified slightly to achieve both the effects of Lasso and forward stagewise regression.

when we use it

**how we use it:**
If we go to the sklearn [docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html), we will find:

*sklearn.linear_model.Lars(fit_intercept=True, verbose=False, normalize=True, precompute=’auto’, n_nonzero_coefs=500, eps=2.220446049250313e-16, copy_X=True, fit_path=True, positive=False)*

no arguments are required for lars, but you can:

**fit_intercept**: default = False
Takes boolian values. Default value is false, meaning no intercept will be calculated or used in calculations. Data is expected to be centred.

**normalize**: default= True
normalizes data when fit_intercept is set to true. For standardized data, use sklearn.preprocessing.StandardScaler before calling lars with normalize = False.

**precompute**: True, False, 'auto', array(gram matrix)

Gram matrix is helpful for determining the linear independance, where the set of vectors are linearly independant if and only if the determinnat of Gram matric is non-zero(source: Horn, Roger A.; Johnson, Charles R. (2013). "7.2 Characterizations and Properties". Matrix Analysis (Second Edition). Cambridge University Press. ISBN 978-0-521-83940-2).

**copy_X**: default = True
Works on a copy of X. If set to false, the original datset may be overwritten.

**fit_path**: default = True
copies the entire path to coeff_path. Setting value to false helps speed up process for large datasets, especially with a small *alpha*.

**Methods**

fit(X, y)
fits the model using given X, y training data

predict(X)
returns prediction values based on the model

score(X, y)	
returns r2 score of prediction
