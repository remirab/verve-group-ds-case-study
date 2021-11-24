# Verve Group DS case study
My approach towards Verve Group data science case study

## Problem statement overview:
A couple of potential problems have been identified that need to be addressed before building and training any machine learning model. In terms of potential problems I would like to give an overview of all of the features to make sure that we are on the same page:
All of the following problems were identified:

* Imbalanced / Skewed classes in features
* Missing / NaN values
* Categorical features
* Non-standard numerical feature
* Unbounded numerical feature

|                        | Imbalanced | Missing | Categorical | Non-std | Unbound |
| :--------------------: | :--------: | :-----: | :---------: | :-----: | :-----: |
| `device_name`          |            |    X    |             |         |         |
| `app_category`         |      X     |    X    |      X      |         |         |
| `ad_category`          |      X     |    X    |      X      |         |         |
| `interaction_with_app` |      X     |         |             |    X    |    X    |
| `click`                |      X     |         |      X      |         |         |
| `gender`               |      X     |         |      X      |         |         |

We are mostly dealing with categorical features that suffer from either being imbalanced or being skewed in some classes. The number of missing values in the feature `device_name` is near 60% and the feature itself is provided in string format. We will further discuss this feature later. The feature `interaction_with_app` is the only numerical feature in nature but still suffers from both being unbounded and being non-standard. The target itself `gender` is highly suffering from positive skewness. I would suggest to analyze our features considering the target that we try to classify to have an intuition model selection.

## Feature engineering & importance:
* The most important feature could be the `app_category` in my opinion. To illustrate, we can imagine that either `male` or `female` targets would get interested in different app categories. For example, `male` targets are mainly into `Sports` or `Automotive` app categories, on the other hand, `female` targets are interested in `Fashion` or `Health` app categories. 

* The same intuition perfectly applies to the `ad_category` feature as well. 

* Although, the industry suggests dropping features with a high volume of missing values near to sixty percent, in our case I believe that the feature `device_name` with near 60% missing value could be very useful due largely to our target feature which is `gender`. We can utilize NLP methods to extract the user's name out of the raw feature and based on the results, transform it into a new categorical feature like `gender_based_on_username` that can be either `male` or `female` or `unisex`. We will discuss how to deal with categorical features later.

* Besides, we can extract `device_type` from `device_name` with the same approach described above. In this case, `device_type` could be categorized as either `tablet` or `phone`. 

* Considering our newly engineered features `gender_based_on_username`, `app_category` and `ad_category`, the feature `click` can be considered interesting, mainly because these four altogether, can illustrate a pattern of interest in our dataset, which the model can learn from. I do understand the importance of getting a click in the AdTech domain, however, considering our target which is `gender` the importance of the `click` feature in this model would be the last one.

* Considering our target, at first sight, the feature `interaction_with_app` seems to be highly unrelated to the target. Although being `male` or `female` seems to be irrelevant to the number of time users spend on mobile apps, this feature alongside others like `app_category` may introduce a special pattern. The amount of time spend regarding the app category illustrates a behavioral pattern. In addition, this feature suffers from being unbound, meaning we can not simply bound this numerical feature into a range. Consequently, we need to determine its upper bound, and standardizing it would be difficult. By contrast, we can benefit from **Bagging** to categorize this feature into categories like `0-10`, `10-20`, `20-30`, ..., `80-90`, `90-100` and `>100`.

* Based on the nature of the `user_id` column, there is a possibility to replace some missing values in the `device_name` feature. We just need to group by our dataset on the `user_id` column and considering the records of the same `user_id` we can replace the missing `device_name` from the records with the same `user_id`.

* The same approach applies to the missing values of the `app_category` feature using the `app_id` column.

* To deal with categorical features, I would suggest using **Target Encoding**. Target Encoding is defined as the process in which:
    > Features are replaced with a blend of the posterior probability of the target given a particular categorical value and the prior probability of the target over all the training data.

* Finally our features would be like:
    * `app_category_encoded`
    * `ad_category_encoded`
    * `gender_based_on_username_encoded`
    * `device_type_encoded`
    * `click_binary_encoded`
    * `interaction_with_app_encoded`
    * `gender_binary_encoded`

## Model selection:
In terms of model selection, considering the high volume of missing values in almost all features and dealing with imbalanced classes, I would suggest using **Random Forest** for the first choice.

* Advantages:
  * It has an effective method for estimating missing values and maintains evaluation metrics when a large proportion of the data are missing.
  * It has methods for balancing errors in datasets where classes are imbalanced.
  * It uses bagging and features randomness when building each tree to try to create an uncorrelated forest of trees.
  * Since averaging of decision trees takes place you can expect less overfitting and less getting stuck with local minima.
    
* Disadvantages:
  * However, some algorithms are still more complex than others when it comes to optimization and parameters. Although random forests are known to work well without too much optimization they still have lots of hyper-parameters that can be adjusted.
  * Random forest models are not all that interpretable; they are like black boxes.

## Model evaluation:
* We need to consider that `accuracy` is not a proper metric for model evaluation in case of highly imbalanced data classes in features. Better choices would be:
    * Precision/Specificity
    * Recall/Sensitivity
    * F1 score
    * ROC
