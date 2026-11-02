# 🏨 Hotel Booking Cancellation Predictor

A machine learning app that predicts whether a hotel booking will be **cancelled**, and recommends what the hotel should do about it.

**🔗 Live app:** `https://YOUR-APP-NAME.streamlit.app` *(replace with your deployed link)*

![App screenshot](images/app_screenshot.png)

---

## 1. Business Problem
Hotels lose revenue when guests cancel, especially at the last minute, because the room often stays empty.
About **29% of bookings** in our data were cancelled.

**Goal:** Predict, at the time of booking, how likely a guest is to cancel, so the hotel can act early. For example, it can:
- ask for a deposit
- send reminders
- offer incentives
- plan safe overbooking

## 2. Dataset
- **Source:** [Hotel Reservations Dataset, Kaggle](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)
- **Size:** 36,275 bookings × 19 columns (2017–2018)
- **Target:** `booking_status` (Canceled / Not_Canceled)
- **Key features:**
  - lead time
  - room price
  - number of special requests
  - market segment (Online, Offline, Corporate…)
  - arrival month
  - meal plan, room type
  - repeat guest, previous cancellations
  - number of guests and nights

## 3. Methodology
1. **Data cleaning**
   - Dropped `Booking_ID`.
   - Removed **10,275 duplicate rows**, so the test score is honest.
   - Removed invalid dates (29 Feb 2018), zero-night stays, zero prices in paid segments and extreme outliers.
   - Final dataset: **25,803 bookings**. There were no missing values.
2. **Exploratory Data Analysis:** Cancellation rate by lead time, special requests, market segment, price, month, and repeat guests. Also a correlation heatmap.
3. **Preparation**
   - Numeric features scaled with `StandardScaler`.
   - Categorical features one-hot encoded.
   - 80/20 stratified train-test split.
4. **Modelling:** Compared Logistic Regression, Decision Tree and Random Forest. All three use `class_weight="balanced"` because the classes are imbalanced.
5. **Evaluation:** Accuracy, Precision, Recall, F1 and ROC-AUC, plus a confusion matrix and feature importance.
6. **Deployment:** Saved the best pipeline (`hotel_cancellation_model.pkl`) and built a Streamlit app.

## 4. Model Used and Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.775 | 0.583 | 0.769 | 0.663 | 0.858 |
| Decision Tree | 0.828 | 0.681 | 0.757 | 0.717 | 0.885 |
| **Random Forest (selected)** | **0.849** | **0.723** | **0.772** | **0.747** | **0.913** |

**Random Forest** was chosen because it has the highest F1 score and ROC-AUC. On 5,161 unseen test bookings, it:
- predicts correctly **85%** of the time
- catches **77%** of actual cancellations

## 5. Business Insights

| Finding | Recommendation |
|---|---|
| Bookings made **180+ days ahead** cancel ~70% of the time, compared with ~14% for bookings within 30 days | Require partial prepayment or stricter cancellation terms for early bookings |
| Guests with **special requests** rarely cancel (0 requests: 41%; 3+: 0%) | Encourage requests during booking (view, pickup, etc.) to build commitment |
| **Online** bookings have the highest cancellation rate (34%) | Offer cheaper non-refundable online rates and send automated reminders |
| **Repeat guests** almost never cancel (1%) | Invest in a loyalty programme |
| Cancelled bookings have **higher prices** (median €110 vs €96) | Use price-match or value-add offers |
| Cancellations peak for **July–August** arrivals | Plan overbooking and deposit rules by season |

**Top predictors** (feature importance): lead time, special requests, room price, arrival month and online booking channel.

## 6. How the App Works
1. A manager enters the booking details: guests, nights, lead time, month, channel, price, meal plan, room type, requests and guest history.
2. The app shows:
   - the **cancellation probability** and **risk level** (Low < 40% ≤ Medium < 70% ≤ High)
   - the **expected revenue at risk**
   - a **recommended business action** for that risk level
   - a short **"Why this result?"** explanation

## 7. How to Run the App Locally
```bash
git clone https://github.com/YOUR-USERNAME/hotel-cancellation-prediction.git
cd hotel-cancellation-prediction
pip install -r requirements.txt
streamlit run app.py
```
The app opens at `http://localhost:8501`.

To re-train the model, open `Hotel_Cancellation_Prediction.ipynb` in Jupyter and run all cells.

## 8. Repository Structure
```
├── app.py                              # Streamlit app
├── Hotel_Cancellation_Prediction.ipynb # Cleaning, EDA, modelling, evaluation
├── hotel_cancellation_model.pkl        # Trained Random Forest pipeline
├── model_metrics.json                  # Test-set scores shown in the app
├── Hotel Reservations.csv              # Dataset
├── requirements.txt                    # Python libraries
├── images/                             # Charts and app screenshot
└── README.md
```

## 9. Limitations and Future Work
- Data comes from one hotel group, 2017–2018. The model should be retrained with recent data.
- The reasons for cancelling are not recorded in the data.
- Future work:
  - try gradient boosting and tune hyperparameters
  - add booking-date features
  - connect the app to the hotel's live booking system

## 10. Team
*(Add the names of all 5 group members here)*
