# Prediction and Analysis of S&amp;P500

## Table of Contents
  * [About](#about)
  * [Contributors](#contributors)
  * [Discussion](#discussion)
    * [Prediction of S&amp;P500 using XGBoost with 10 features](#prediction-of-sandp500-using-xgboost-with-10-features)
      * [Validation of the 10-feature XGBoost Model](#validation-of-the-10-feature-xgboost-model)
      * [Feature optimization and validation of feature optimized models](#feature-optimization-and-validation-of-feature-optimized-models)
    * [Interpretation of XGBoost Feature Importance and Rule Derivation to Find Entry Points](#interpretation-of-xgboost-feature-importance-and-rule-derivation-to-find-entry-points)
      * [Challenges of Interpreting XGBoost Feature Importance](#challenges-of-interpreting-xgboost-feature-importance)
      * [Rule Derivations](#rule-derivations)
      * [Summary on Entry Point Derivation](#summary-on-entry-point-derivation)
  * [Outcome](#outcome)
  * [Future Work](#future-work)
  * [References](#references)
  

---

## About 
This objective of this project is to develop a predictive model for the S&amp;P500 stock index using economic indicators from FRED, Federal Reserve Economic Data, an online database consisting of hundreds of thousands of publicly available economic data time series, to find entry points and indicators to purchase, sell or hold in the stock market to maximise returns.


Indicators used from FRED for prediction (subject to expansion and change):
1. GDP (Gross domestic product)
2. CPIAUCSL (Get Consumer Price Index for All Urban Consumers: All Items in U.S. City Average)
3. UNRATE (Unemployment Rate)
4. AMTMNO (Manufacturers' New Orders: Total Manufacturing)
5. POILWTIUSDM (Global price of WTI Crude)
6. AAA10Y (Moody's Seasoned Aaa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity)
7. CSCICP03USM665S (Consumer Opinion Surveys: Confidence Indicators: Composite Indicators: OECD Indicator for the United States)
8. T10Y3M (10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity)
9. VIXCLS (CBOE Volatility Index: VIX)
10. BSCICP03USM665S (Business Tendency Surveys for Manufacturing: Confidence Indicators: Composite Indicators: OECD Indicator for the United States)


ML techniques used for prediction (subject to expansion and change):
- XGBoost (Extreme Gradient Boosted Decision Tree)

---

## Contributors
- John (@jpl12345)

---
## Discussion

### Prediction of SANDP500 using XGBoost with 10 features
10 indicators from FRED were used to [train XGBoost](/ml_for_spy_v2_ffill_daily_xgboost.ipynb): GDP, CPIAUCSL, UNRATE, AMTMNO, POILWTIUSDM, AAA10Y, CSCICP03USM665S, T10Y3M, VIXCLS and BSCICP03USM665S. The indicators that were chosen are related to the US Treasury's yield curve (AAA10Y, T10Y3M) which has been [studied extensively as a predictor of recession](https://www.chicagofed.org/publications/chicago-fed-letter/2018/404), as well as GDP [with dips usually associated with recession](https://www.cnbc.com/2022/07/28/gdp-dips-sparking-recession-fears-whats-really-happening.html), CPI [which is an indicator of inflation](https://www.rba.gov.au/education/resources/explainers/inflation-and-its-measurement.html#:~:text=The%20most%20well%2Dknown%20indicator,and%20services%20consumed%20by%20households.), UNRATE [(unemployment rate) which usually sharply increases during recessions](https://www.bls.gov/opub/mlr/2021/article/unemployment-rises-in-2020-as-the-country-battles-the-covid-19-pandemic.htm), consumer confidence indicators, as well as business confidence indicators, and VIXCLS which measures the market's uncertainty.

Federal Reserve Economic Data (FRED) was chosen as the main data source because of its vast amount of reputable publicly available indicators easily accessible through FRED's API without a paywall. 

For training the XGBoost model, data was split into random train and test subsets - 80% of the data was used for training, and the remaining 20% was used for validation. XGBoost was able to predict S&amp;P500 on a test data set with a Root Mean Squared Error (RMSE): 34.422484825156396 and R-squared (R2) Score: 0.9988822689809329.

![Predicted VS SPY for test dataset](/figures/predictedvsSPY_Testdata.png)

**Figure 1: Predicted vs Actual for the test dataset.** As the test dataset was not from a contiguous time period due to the usage of train_test_split, it is difficult to see the predicted vs actual performance for the test dataset. To better visualise the predicted vs actual performance of the model, please refer to figure 2.



![Predicted VS SPY for test and train dataset](/figures/predictedvsSPY_inclTrain.png)

**Figure 2: Predicted vs Actual for the train + test dataset.**

Different indicators had different update periods. For instance, CPIAUCSL was updated monthly, GDP was updated quarterly whereas VIXCLS was updated daily. Therefore, forward fill using the most recently updated value was used to fill values in between update dates till the last existing date in ‘SPY.csv’. This was done as daily S&P500 prices were predicted, and missing values in the indicators between dates would adversely affect the prediction. However, the decision to forward fill values between dates may have affected the feature importance. As GDP was updated the least frequently compared to other indicators, quarterly gaps between updates may have made it less important for predicting daily S&P500 prices. Conversely the 3 indicators which were updated daily, T10Y3M, VIXCLS and AAA10Y were the top 3 most important features, which might have been because of their daily update frequency.

![Feature Importance](/figures/xgboost_f_scores.png)

**Figure 4: F-score for the 10 features used.** A higher f-score indicates higher feature importance. 

Unfortunately, most of the data available on FRED is not updated daily, instead most of the data is either updated yearly or monthly as shown below. As such, future work can explore either predicting monthly values of SPY, or alternatively explore daily indicators exclusively.

![Fred data frequency](/figures/Capture_of_FRED_data_updatefreq.PNG)

**Figure 5: Screenshot of the update frequency of available datasets on FRED**


Of the 10 indicators, VIXCLS was the most important. VIX measures market expectation of near term volatility conveyed by stock index option prices. Large values of VIX indicate high volatility, and low values indicate low volatility. And during recession periods, such as the 2008 financial crisis, and the 2020 COVID stock market crash VIX spiked sharply. As such, sharp spikes in VIX could indicate a buying opportunity. This can be seen in the figure 6.

The 2nd most important feature was AAA10Y/AAAvstreasury, Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity is a financial indicator that compares the yield on corporate bonds to the yield on 10-year U.S. Treasury bonds. It is often used as a measure of the risk premium investors demand for investing in corporate bonds compared to the relatively risk-free Treasury bonds. Positive value indicates that corporate bonds are offering a higher yield than Treasury bonds, reflecting the additional risk associated with corporate debt. A negative value suggests that corporate bonds are offering a lower yield compared to Treasury bonds, indicating a lower perceived risk in the corporate bond market. Large spikes in AAA10Y coalesce with dips in SPY, as seen in figure 6.

The 3rd most important feature was T10Y3M, 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity, a measure of the yield curve slope. A positive value indicates a positive yield curve slope, also known as a normal yield curve, where long-term interest rates are higher than short-term interest rates. This is generally considered a healthy and normal economic environment. Conversely, a negative value suggests an inverted yield curve, where short-term interest rates are higher than long-term interest rates. An inverted yield curve is often seen as a potential indicator of an economic recession or market uncertainty. The relationship between sharp dips in T10Y3M and SPY are not as clear, because not all sharp dips into the negatives coalesce with sharp dips or recessionary periods in SPY.

![top3vsSPY](/figures/predictedvsSPY_inclTrain_withsubplots_fortop3.png)

**Figure 6: Daily movement of SPY from 1-Jan-1992 to 30-June-2022 (both predicted and actual) compared to VIXCLS, AAA10Y/AAAvstreasury and T10Y3M**

<br>
<br>

### Validation of the 10-feature XGBoost Model

![validation_10f_xgb](figures/validationfrom2023_predictedvsSPY_inclTrain.png)
**Figure 7: Comparison of predicted vs actual for 2023 SPY values.** The vertical red line marks when the model was trained.

<br>

### Feature optimization and validation of feature optimized models
The goal of feature optimization was to reduce the number of features while preserving an acceptable level of performance (R2>0.99 on test data, and RMSE should be comparable to the 10-feature model).

Using only the top 3 features for the model was insufficient, as the R2 value did not meet the threshold.

Using the top 4 features nearly met the threshold of 0.99. But in figure 8, there are some years where the predicted value sharply deviated from the actual value. Therefore using 4 features is insufficient. 

![predictedvsSPY_inclTrain-4Features](figures/predictedvsSPY_inclTrain-4Features.png)
**Figure 8: Comparison of predicted vs actual for train and test data using the 4 feature model.**

The R2 value using the top 5 features (0.9989) met the threshold of 0.99. No sharp spikes between actual and predicted values were observed in figure 9. Moreover, the RMSE of using 5 features (36.9974) is only slightly worse than using 10 features (34.4225).

![predictedvsSPY_inclTrain-5Features](figures/predictedvsSPY_inclTrain_withsubplots_for_top5.png)
**Figure 9: Comparison of predicted vs actual for train and test data using the 4 feature model.**

Unfortunately when validating the model, using the top 5 features produces an unacceptable error shown in figure 10. This trend was observed for the models using the top 5-7 features, despite the R2 value of the test data meeting the threshold. Only after using equal or more than 8 features, was the level of performance compared to the orignal 10 feature model comparable, as shown in figure 11. Therefore, the number of features used can be reduced to 8, which consists of: 'vix', 'AAAvstreasury', 't10y3m', 'unemp_rate', 'oil_prices', 'bus_confidence','mno', 'cpi'

![validation_5f_xgb](figures/validationfrom2023_predictedvsSPY_inclTrain_for5featuremodel.png)
**Figure 10: Comparison of predicted vs actual for 2023 SPY values for the 5 feature model.** The vertical red line marks when the model was trained.


![validation_8f_xgb](figures/validationfrom2023_predictedvsSPY_inclTrain_for8featuremodel.png)
**Figure 11: Comparison of predicted vs actual for 2023 SPY values for the 8 feature model.** The vertical red line marks when the model was trained.

---

### Interpretation of XGBoost Feature Importance and Rule Derivation to Find Entry Points
In the next phase of the project, the feature importance generated by the XGBoost model were used to create rules are derived to identify entry points for buying SPY. 
The XGBoost model assigns importance scores to different features based on their contribution to predicting SPY's price movements. These scores provide insights into which economic indicators are most influential in determining when SPY might experience local bottoms, indicating potential buying opportunities. The higher the importance score, the stronger the feature's impact on predicting price changes. In order for these rules to hold, a critical assumption was made that at some point in the future, the value of SPY will be higher than what it currently is now.

### Challenges of Interpreting XGBoost Feature Importance
XGBoost is capable of capturing nonlinear relationships between features and target variables. Interpreting the feature importance scores in the presence of complex nonlinear interactions can be challenging, as the impact of a feature might vary across different ranges of its values. Moreover, XGBoost can capture interactions between features even if they are not explicitly defined. The resulting feature importance scores may not reveal the specific combinations of features that contribute to accurate predictions, making it difficult to interpret the model's decision-making process.
XGBoost models consist of an ensemble of decision trees, each contributing to the final prediction. Interpreting individual trees independently might not provide a comprehensive understanding of the model's behavior, as predictions are made based on the combined output of multiple trees. As such. converting a complex tree structure into interpretable rules often requires simplification. This simplification can lead to ambiguity in rule interpretation and the potential loss of nuanced patterns.


### Rule derivations
To evaluate the effectiveness of each rule the following function will be used:
```
def eval_spy_stratergy(boy_year, eoy_year, end_of_year_price, dataframe_containing_spy_buying_opp, name_of_col_with_price, is_date_used_asIndex):
    print("=====")
    print(f'The end of year price for {eoy_year} is ${end_of_year_price}')
    avg_price_of_spy_inPortfolio = dataframe_containing_spy_buying_opp[f'{name_of_col_with_price}'].mean()
    print(f'The average price of SPY in your portfolio would be: ${avg_price_of_spy_inPortfolio}')
    print(f'The number of buying opportunites from {boy_year} to {eoy_year} are: {len(dataframe_containing_spy_buying_opp.index)}')
    profit = ((end_of_year_price-avg_price_of_spy_inPortfolio)/avg_price_of_spy_inPortfolio)*100
    print(f'% profit/loss = {profit:.2f}%')
    if (is_date_used_asIndex==True):
        years_with_buying_opportunities = dataframe_containing_spy_buying_opp.index.year.unique().tolist()
        num_months_with_buying_opportunities = dataframe_containing_spy_buying_opp.index.to_period('M').nunique()
        print(f'{len(years_with_buying_opportunities)} years and {num_months_with_buying_opportunities} months with buying opportunities: {years_with_buying_opportunities}')
        
        
    else:
        years_with_buying_opportunities = dataframe_containing_spy_buying_opp['Date'].dt.year.unique().tolist()
        num_months_with_buying_opportunities = dataframe_containing_spy_buying_opp.set_index('Date').index.to_period('M').nunique()
        print(f'{len(years_with_buying_opportunities)} years and {num_months_with_buying_opportunities} months with buying opportunities: {years_with_buying_opportunities}')
    
    print("=====")
    return profit, years_with_buying_opportunities, num_months_with_buying_opportunities


```
The function calculates the average cost of SPY in a portfolio if the rule were to be applied, and it would compare the average cost of SPY compared to a fixed endpoint value on 30 December 2022 and calculates the profit. Additionally, it would also indicate the number of years and months with buying opportunities.

The perfect rule would yield the highest profit, and the maximum number of buying opportunities across all the years.

Naive Rule of Buying SPY at the End of Every Month

This rule involves buying SPY at the end of every month and calculating the profit or loss based on the difference between the end-of-year price for a given year and the average price of SPY in the portfolio. This simple rule assumes a consistent investment strategy and serves as the benchmark for other rules to beat.
The outcome of using the naïve rule is: 
```
=====
The end of year price for 2022 is $3839.5
The average price of SPY in your portfolio would be: $1868.1272713955966
The number of buying opportunites from 2000 to 2022 are: 275
% profit/loss = 105.53%
23 years and 275 months with buying opportunities: [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
=====
```
<br>

**Semi-Intelligent Rule 1 - Buy When Close Price is Below Moving Average - 1.96 * SD**

In this rule, the moving average and standard deviation of the close price of SPY over a rolling 365-day window was calculated. If the close price of SPY falls below the moving average minus 1.96 times the standard deviation, consider it a buying opportunity. This rule attempts to capture points where the price of SPY is significantly below its average and assumes a level of market volatility.

The outcome of using semi-intelligent rule 1 is:
```
=====
The end of year price for 2020 is $3839.5
The average price of SPY in your portfolio would be: $1287.4026194807059
The number of buying opportunites from 2000 to 2020 are: 313
% profit/loss = 198.24%
9 years and 29 months with buying opportunities: [2000, 2001, 2002, 2008, 2009, 2016, 2018, 2020, 2022]
=====
```
![semiInt1](figures/entrypoints/semi_int_rule1.png)
Figure 12: Visualization of the entry points using semi-intelligent rule 1 in red. 
<br>

**Semi-Intelligent Rule 2 - Buy When Close Price is Below Moving Average - 2.576 * SD**

Similar to Rule 2, this rule involves calculating the moving average and standard deviation of the close price of SPY over a rolling 365-day window. However, it uses a more stringent criterion, where a buying opportunity if the close price of SPY falls below the moving average minus 2.576 times the standard deviation. This rule aims to capture even more extreme market movements.

The outcome of semi-intelligent rule 2 is:
```
=====
The end of year price for 2020 is $3839.5
The average price of SPY in your portfolio would be: $1088.772617969698
The number of buying opportunites from 2000 to 2020 are: 103
% profit/loss = 252.64%
6 years and 16 months with buying opportunities: [2001, 2002, 2008, 2016, 2020, 2022]
=====
```
![semiInt2](figures/entrypoints/semi_int_rule2.png)
Figure 13: Visualization of the entry points using semi-intelligent rule 2 in red. 

<br>

There is a tradeoff between the number of years and months with buying opportunities and maximum profit with the standard deviation value chosen as shown in figure 14. Personally, I believe using rule 1 (SD=1.96) yields an acceptable tradeoff between profit and number of buying opportunities.

![tradeoff1](figures/entrypoints/semi_int_rule_tradeoff.png)
Figure 14: Profit/Loss, Number of Months, and Number of Years with Buying Opportunities vs. Standard Deviation Value. 

#### Predictive Rules Based on Feature Importance from XGBoost

This set of rules involves using feature importance from an XGBoost model to predict entry points for buying SPY. The rules are derived from analyzing relationships between SPY's close price and other relevant features.

**Predictive Rule 1: Buy When AAAvsTreasury and VIX Are High**

For this rule, the moving average and standard deviation of AAAvsTreasury and VIX was calculated over a rolling 365-day window. If the value of AAAvsTreasury is above the moving average plus 1.96 times the standard deviation, and the value of VIX is also above its moving average plus 1.96 times the standard deviation, consider it a buying opportunity. This rule assumes that high values of these indicators may suggest a favorable buying point.

The outcome of predictive rule 1 is:
```
=====
The end of year price for 2022 is $3839.5
The average price of SPY in your portfolio would be: $1491.9916801096124
The number of buying opportunites from 2000 to 2022 are: 214
% profit/loss = 157.34%
9 years and 27 months with buying opportunities: [2000, 2001, 2007, 2008, 2011, 2015, 2016, 2019, 2020]
=====
```
![predRule1](figures/entrypoints/predictive_rule1.png)
Figure 15: Visualization of the entry points using predictive rule 1 in red. 

<br>

**Predictive Rule 2: Rule Incorporating Rule 1 and when Unemployment Rate is high and Oil Prices are low**

This rule builds on the previous rule and includes additional conditions based on unemployment rate and oil prices. The moving average and standard deviation of the unemployment rate and oil prices was calculated over shorter rolling windows (12 months). The rule requires that both the unemployment rate and oil prices be below their moving averages plus 1.96 times the standard deviation. The intention is to identify buying opportunities when certain economic indicators indicate a negative outlook.

The outcome of predictive rule 2 is:
```
=====
The end of year price for 2022 is $3839.5
The average price of SPY in your portfolio would be: $1459.3629531860352
The number of buying opportunites from 2000 to 2022 are: 88
% profit/loss = 163.09%
4 years and 7 months with buying opportunities: [2001, 2007, 2008, 2020]
=====
```
![predRule2](figures/entrypoints/predictive_rule2.png)
Figure 16: Visualization of the entry points using predictive rule 2 in red. 

<br>

### Summary on Entry Point Derivation

Although both predictive rules did not surpass the semi-intelligent rule based on a standard deviation of 1.96 or greater, they still yielded higher profits compared to the naive rule. This observation emphasizes the value of leveraging machine learning models to inform trading decisions, even when the model's rules do not outperform simpler strategies. The predictive rules, by capturing nuanced relationships between economic indicators and market movements, showcased the potential for more sophisticated entry point identification.

As part of future work, enhancing the interpretability of the XGBoost model's decisions could offer insights into the intricate relationships between features and target variables. Techniques such as SHAP (SHapley Additive exPlanations) values and partial dependence plots could be explored to provide a clearer understanding of the model's decision-making process. By visualizing the impact of individual features on predictions, these techniques could facilitate the derivation of more effective predictive rules. This approach aligns with the broader goal of refining trading strategies and uncovering deeper insights into the underlying dynamics of the stock market.

In conclusion, the exploration of predictive rule derivations, guided by feature importance from the XGBoost model, showcased the potential for leveraging complex relationships between economic indicators to identify optimal entry points for purchasing SPY. While certain rules did not surpass simpler strategies, they contributed to my overall understanding of market dynamics and the potential for sophisticated trading strategies. 

<br>

## Outcome

Using XGBoost was able to predict S&amp;P500 with 10 features on a test data set with a Root Mean Squared Error (RMSE): 34.422484825156396 and R-squared (R2) Score: 0.9988822689809329. After feature optimization, the number of features used can be reduced to 8, which consists of: 'vix', 'AAAvstreasury', 't10y3m', 'unemp_rate', 'oil_prices', 'bus_confidence','mno', 'cpi'. Addtionally rules to derive entry points to maximise profit on SPY in a portfolio context were derived using the XGBoost model.


---

## Future Work
Model-related:
1. Fine-tuning and optimizing the XGBoost model
2. Exploring other models such as LSTM
3. SHAP (SHapley Additive exPlanations) values and partial dependence plots exploration to derive improved predictive rules

Feature selection and optimization:
1. Exploring other indicators within FRED and beyond FRED.
2. Exploring the effect of using only daily indicators instead of the current mix of daily, monthly and quarterly indicators.


---

## References
https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=3768&context=cmc_theses
https://github.com/eggrollofchaos/i_spy_etf_price_predictions
https://github.com/snsten/SP500_Time-Series_Forecasting/tree/master
http://www.econ2.jhu.edu/people/Duffee/jf_spreads.pdf
https://www.sciencedirect.com/science/article/abs/pii/S0957417421010988
