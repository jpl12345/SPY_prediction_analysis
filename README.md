# Prediction and Analysis of S&amp;P500

## Table of Contents
  * [About](#about)
  * [Contributors](#contributors)
  * [Discussion](#discussion)
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

### For the [prediction of daily S&amp;P500 index using XGBoost](/ml_for_spy_v2_ffill_daily_xgboost.ipynb)
10 indicators from FRED were used to train XGBoost: GDP, CPIAUCSL, UNRATE, AMTMNO, POILWTIUSDM, AAA10Y, CSCICP03USM665S, T10Y3M, VIXCLS and BSCICP03USM665S. The indicators that were chosen are related to the US Treasury's yield curve (AAA10Y, T10Y3M) which has been [studied extensively as a predictor of recession](https://www.chicagofed.org/publications/chicago-fed-letter/2018/404), as well as GDP [with dips usually associated with recession](https://www.cnbc.com/2022/07/28/gdp-dips-sparking-recession-fears-whats-really-happening.html), CPI [which is an indicator of inflation](https://www.rba.gov.au/education/resources/explainers/inflation-and-its-measurement.html#:~:text=The%20most%20well%2Dknown%20indicator,and%20services%20consumed%20by%20households.), UNRATE [(unemployment rate) which usually sharply increases during recessions](https://www.bls.gov/opub/mlr/2021/article/unemployment-rises-in-2020-as-the-country-battles-the-covid-19-pandemic.htm), consumer confidence indicators, as well as business confidence indicators, and VIXCLS which measures the market's uncertainty.

Federal Reserve Economic Data (FRED) was chosen as the main data source because of its vast amount of publicly available indicators which can be easily through FRED's API without a paywall. 

For training the XGBoost model, data was split into random train and test subsets - 80% of the data was used for training, and the remaining 20% was used for validation. XGBoost was able to predict S&amp;P500 on a test data set with a Root Mean Squared Error (RMSE): 34.422484825156396 and R-squared (R2) Score: 0.9988822689809329.

![Predicted VS SPY for test dataset](/figures/predictedvsSPY_Testdata.png)
![Predicted VS SPY for test and train dataset](/figures/predictedvsSPY_inclTrain.png)

Different indicators had different update periods. For instance, CPIAUCSL was updated monthly, GDP was updated quarterly whereas VIXCLS was updated daily. Therefore, forward fill using the most recently updated value was used to fill values in between update dates till the last existing date in ‘SPY.csv’. This was done as daily S&P500 prices were predicted, and missing values in the indicators between dates would adversely affect the prediction. However, the decision to forward fill values between dates may have affected the feature importance. As GDP was updated the least frequently compared to other indicators, quarterly gaps between updates may have made it less important for predicting daily S&P500 prices. Conversely the 3 indicators which were updated daily, T10Y3M, VIXCLS and AAA10Y were the top 3 most important features, which might have been because of their daily update frequency.

![Feature Importance](/figures/xgboost_f_scores.png)

Unfortunately, most of the data available on FRED is not updated daily, instead most of the data is either updated yearly or monthly as shown below. As such, future work can explore either predicting monthly values of SPY, or alternatively explore daily indicators exclusively.

![Fred data frequency](/figures/Capture_of_FRED_data_updatefreq.PNG)


Of the 10 indicators, VIXCLS was the most important. VIX measures market expectation of near term volatility conveyed by stock index option prices. Large values of VIX indicate high volatility, and low values indicate low volatility. And during recession periods, such as the 2008 financial crisis, and the 2020 COVID stock market crash VIX spiked sharply. As such, sharp spikes in VIX could indicate a buying opportunity. This can be seen in the figure 4.

The 2nd most important feature was AAA10Y/AAAvstreasury, Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity is a financial indicator that compares the yield on corporate bonds to the yield on 10-year U.S. Treasury bonds. It is often used as a measure of the risk premium investors demand for investing in corporate bonds compared to the relatively risk-free Treasury bonds. Positive value indicates that corporate bonds are offering a higher yield than Treasury bonds, reflecting the additional risk associated with corporate debt. A negative value suggests that corporate bonds are offering a lower yield compared to Treasury bonds, indicating a lower perceived risk in the corporate bond market. Large spikes in AAA10Y coalesce with dips in SPY, as seen in figure 4.

The 3rd most important feature was T10Y3M, 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity, a measure of the yield curve slope. A positive value indicates a positive yield curve slope, also known as a normal yield curve, where long-term interest rates are higher than short-term interest rates. This is generally considered a healthy and normal economic environment. Conversely, a negative value suggests an inverted yield curve, where short-term interest rates are higher than long-term interest rates. An inverted yield curve is often seen as a potential indicator of an economic recession or market uncertainty. The relationship between sharp dips in T10Y3M and SPY are not as clear, because not all sharp dips into the negatives coalesce with sharp dips or recessionary periods in SPY.

![top3vsSPY](/figures/predictedvsSPY_inclTrain_withsubplots_fortop3.png)
**Figure 4: Daily movement of SPY from 1-Jan-1992 to 30-June-2022 (both predicted and actual) compared to VIXCLS, AAA10Y/AAAvstreasury and T10Y3M**








---

## Outcome

Using XGBoost was able to predict S&amp;P500 on a test data set with a Root Mean Squared Error (RMSE): 34.422484825156396 and R-squared (R2) Score: 0.9988822689809329. Additionally, 3 indicators were identified as important VIXCLS, AAA10Y and T10Y3M.



---

## Future Work
Model-related:
1. Fine-tuning and optimizing the XGBoost model
2. Exploring other models such as LSTM

Feature selection and optimization:
1. Exploring the effect of only keeping the top 3 indicators for the prediction of S&P500
2. Exploring other indicators within FRED and beyond FRED.


---

## References
https://scholarship.claremont.edu/cgi/viewcontent.cgi?article=3768&context=cmc_theses
https://github.com/eggrollofchaos/i_spy_etf_price_predictions
https://github.com/snsten/SP500_Time-Series_Forecasting/tree/master
http://www.econ2.jhu.edu/people/Duffee/jf_spreads.pdf
https://www.sciencedirect.com/science/article/abs/pii/S0957417421010988
