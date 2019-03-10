# 2019-kaggle-elo-top-11-solution
https://www.kaggle.com/c/elo-merchant-category-recommendation

This is the 11th place solution writeup.
Hello, I would like to congratulate everyone for the efforts they put into this competition. And ELO for hosting such a great competition with interesting data and hair pulling outliers :D

I would like to give special thanks to my teammates @Zakaria EL Mesaoudi @Nabil @Lindada for their skills, perseverance and hardwork.

The huge gap between CV and LB did not make it easy for us (and for everyone I imagine). Those that persevered until the end would perfectly understand.

Congrats for all the medalists, you did a tremendous job :D

#### SUMMARY
So basically we were two separate teams that merged a couple of hours before the deadline. Each had had completed a tremendous amount of work, 
and after almost running out of new ideas, a merge was deemed necessary (and fruitful).

#### FEATURE ENGINEERING
Like most public kernels, we constructed aggregation features. Here's a list of the our strongest ones:

1. I refer the Kaggle Rank System Compute Formula（link:[https://www.kaggle.com/progression][4])
    df_data['duration_sqrt_counts'] = df_data['durations']/sqrt(df_data['card_id_counts'])
    df_data['duration_log1p_counts'] = df_data['durations']/log1p(df_data['card_id_counts'])
    df_data['duration_counts'] = df_data['durations']/df_data['card_id_counts']

2. Categorical features: frequence, Maxfrequence, MaxfrequenceRatio

3. card_id/merchant_id/mechant_category_id/city_id (visit sequence to sequence embedding)

4. purchase_amount:hist/new

5. features interactions between hist/new
            df['purchase_amount_ratio_v3'] =                              df['new_purchase_amount_max']/df['hist_purchase_amount_sum']
            df['purchase_amount_diff_v1'] = df['new_purchase_amount_sum']-df['hist_purchase_amount_sum']
            df['purchase_amount_diff_v2'] = df['new_purchase_amount_mean']-df['hist_purchase_amount_mean']
            df['purchase_amount_diff_v3'] = df['new_purchase_amount_max']-df['hist_purchase_amount_max']
            df['purchase_amount_diff_v4'] = df['new_purchase_amount_min']-df['hist_purchase_amount_min']
            df['pa_mlag_ratio'] = df['new_purchase_amount_sum']/(df['month_lag_mean'] - 1)
            df['pa_new_hist_ratio'] = df['new_purchase_amount_sum']/(df['hist_purchase_amount_sum'])
            df['pa_new_hist_mean_ratio'] = df['new_purchase_amount_mean']/(df['hist_purchase_amount_mean'] )
            df['pa_new_hist_min_ratio'] = df['new_purchase_amount_min']/(df['hist_purchase_amount_min'] )
            df['pa_new_hist_max_ratio'] = df['new_purchase_amount_max']/(df['hist_purchase_amount_max'] )
We had two separate feature sets. One with +1000 features and another one with +200 features

At this point our best models scored around: CV 3.642X and LB: 3.688 and CV 3.644X and LB: 3.686

After that, we took the correlation matrix of the +200 feature set and paired each feature with the feature it's the least correlated with. 
Then we applied a bunch of aggregations on each pair and it resulted in pretty strong features.

So we ended up with two feature sets with +1000 features each.

#### FEATURE SELECTION
For feature selection we did some manual feature selection based on the features importance feedback we got from lgb. Then we used this (simple yet effective) method here for some further filtering. What it basically does is remove:

Features with a high percentage of missing values according to a
threshold
Collinear (highly correlated) features
Features with zero importance in a tree-based model
Features with low importance
Features with a single unique value
The CV score got better in both feature sets.

After this, our best models scored around CV:3.639X LB:3.682

#### MODELS
We used different models for training: LightGBM / Catboost / Xgboost / H2oRF / H2oGBM. 
We tried a couple of NN architectures but it didn't work out for us (Check the last section)

#### STACKING
We stacked around 32 models using bayesian regression. Our models were well varied that it yielded a score of CV:3.630X LB :3.675

#### POST PROCESSING
During the last day, we focused on doing some post processing and this is how we cherrypicked our outliers.

we carefully designed a Classifying module that combined four different classifers.

We picked the top 100 outliers predicted from every classifier and then intersected the four sets. This resulted in 21 final outliers.

This gave us a devilish boost on LB. We went from 3.675 to 3.666

#### SUBMISSIONS
Finally we chose:

A model without post processing (Our best stacking sub): CV:3.63019 LB:3.675 and Private LB: 3.599
A model with post processing: LB:3.666 and Private:3.599
STUFF THAT DID NOT WORK
Of course these last two months were not all roses and rainbows. We pulled our hair trying a lot of things and we failed miserably.

Here are the bloopers of our participation :D :

NN. We tried designing different architectures with the main focus on having a simple NN with heavy regularization (BatchNorm and Strong Dropout)
In the middle of the competition, we tried tackling the outliers detection as an anomaly detection problem using AutoEncoders trained only on the non outliers data
We tried PCA for more features. And it didn't work
We tried TSNE. It didn't work
We tried FM and FFM. It did not work
We tried isolation forest. Nope. Did not work.
We had a Ridge-based pairwise ranker that we intended to use for outliers detection but it didn't match with the approach we had.
We tried a lot of weak models in the hope of adding diversity (simple tree-based, linear, svm, etc.). And guess what? It did not work.
Anyway, that was a rough wrap up of all the things we did.

Thank you for sparing the time to read this.

And remember to always trust CV :D
