# Credit-Card-Fraud-Detection
Leveraging V17 With Machine Learning

## My Project Story
Hi, I’m Abimbola, and this is my project, "Credit Card Fraud Detection: Leveraging V17 with Machine Learning." I built this project from scratch to detect fraudulent credit card transactions using the creditcard.csv dataset, and I’m really proud of how it turned out! My main goal was to create a machine learning model that could spot fraud effectively, and I decided to focus on the V17 feature because I discovered it was a key indicator of fraud. I worked on this in Google Colab, following a 1-8 step process that I designed to cover everything from data setup to deployment preparation. After a lot of experimentation, I ended up with a Random Forest model that achieved an AUPRC of 0.880 and a cost of $8,650, which I think is pretty solid for this kind of problem.

I wrote this README myself to share what I did, what I learned, and the results I got. This project was a great learning experience for me, and I hope it shows my skills in data science, machine learning, and problem-solving. Whether you’re a professor, recruiter, or fellow data enthusiast, I’m excited to share my work with you!

### 🎯 What I Set Out to Do
I had a few goals for this project:

- Build a model to detect credit card fraud, with a special focus on the V17 feature that I found to be important.
- Tackle the dataset’s imbalance (only 0.17% of transactions are fraud).
- Minimize costs, where missing a fraud (false negative) costs $500 and a false alarm (false positive) costs $50.
- Make sure my model’s decisions are understandable and transparent.
- Get the model ready for real-world use, like in a live fraud detection system.

### 📊 The Data I Worked With
- Dataset: I used creditcard.csv, which I got from Kaggle.
- Size: It has 284,807 transactions and 31 features.
- Features:
   - Time: Seconds since the first transaction (about 48 hours).
   - V1 to V28: These are anonymized features (I think they’re PCA components).
   - Amount: The transaction amount.
   - Class: The target (0 for non-fraud, 1 for fraud).
- Class Distribution: I found that 284,315 transactions are non-fraud (99.83%), and only 492 are fraud (0.17%), so it’s super imbalanced.

### 🛠️ How I Did It: My Step-by-Step Approach
I broke this project into 8 steps to make sure I covered everything. Here’s what I did in each step, straight from my notebook in Google Colab.

#### Step 1: Getting Started
- What I Did:
   - I imported the libraries I needed, like pandas, numpy, seaborn, and matplotlib, to handle data and make plots.
   - I mounted Google Drive in Colab to load creditcard.csv.
   - I checked the data’s shape, types, and missing values, and looked at the class distribution.
- What I Found:
   - The dataset has 284,807 rows and 31 columns, with no missing values.
   - It’s very imbalanced: 284,315 non-fraud vs. 492 fraud.
 
#### Step 2: Exploring the Data (EDA)
- What I Did:
   - I plotted the time density of transactions to see when fraud happens compared to non-fraud.
   - I looked at time gaps between transactions to spot patterns in fraud.
   - I checked how Amount is distributed for fraud cases.
   - I calculated spike ratios for V1 to V28 to find which features stand out in fraud cases.
- What I Found:
   - V17 had the highest spike ratio (13.23), meaning its values are 13.23 times larger in fraud cases on average. Other features like V14 (10.87) and V12 (8.95) were also important.
   - Fraud often happens at certain times (like late at night) and in quick bursts.
   - Fraud transactions are spread across Amount values, with many being small amounts.
 
#### Step 3: Preparing the Data
- What I Did:
   - I split the data into 80% training (227,846 samples) and 20% testing (56,961 samples), making sure the fraud ratio stayed the same with random_state=42.
   - I added three new features:
      - V17_boost: To highlight extreme V17 values (above the 95th percentile).
      - Time_cycle: To capture 48-hour patterns using sin(2 * π * Time / (48 * 3600)).
      - Log_Amount: To normalize Amount by capping it at the 99th percentile and taking the log.
   - I made sure there were no NaNs after adding these features.
- What I Found:
   - My dataset now had 33 features, and the new features were ready to help my model.
 
#### Step 4: Handling the Imbalance
- What I Did:
   - I oversampled the fraud cases in the training set using V17_boost to decide which ones to pick more often, aiming for 10% of the non-fraud count (22,745 fraud samples). I set random_state=42 for consistency.
   - I combined the oversampled fraud data with the non-fraud data.
   - I checked for NaNs to make sure everything was clean.
- What I Found:
   - I got a balanced training set: 227,451 non-fraud and 22,745 fraud (a 10:1 ratio, much better than the original 578:1).
 
#### Step 5: Building and Tuning Models
- What I Did:
   - I trained four models: Quadratic Discriminant Analysis (QDA), Logistic Regression (LR), Random Forest (RF), and Support Vector Machine (SVM), all with random_state=42.
   - I wrote a function to find the best threshold for each model by maximizing the F1 score, using AUPRC as my main metric.
   - I tested the models on the test set.
- What I Found:
   - Here’s how they performed:
      - QDA: AUPRC 0.492, Threshold 1.000
      - LR: AUPRC 0.740, Threshold 0.989
      - RF: AUPRC 0.880, Threshold 0.560
      - SVM: AUPRC 0.657, Threshold 0.846
   - RF was the best, with a great AUPRC and a balanced threshold.
 
#### Step 6: Evaluating the Models
- What I Did:
   - I set up a cost metric: $500 for each missed fraud (false negative) and $50 for each false alarm (false positive).
   - I calculated the cost for each model on the test set using their best thresholds.
   - I picked the model with the lowest cost.
- What I Found:
   - Costs for each model:
       - QDA: $27,000, AUPRC 0.492
       - LR: $9,400, AUPRC 0.740
       - RF: $8,650, AUPRC 0.880
       - SVM: $12,850, AUPRC 0.657
   - RF had the lowest cost, so I chose it as my final model.
 

#### Step 7: Understanding My Model
- What I Did:
   - I looked at feature importance in my RF model to see which features mattered most.
   - I used SHAP to dig deeper into how features like V17 affect predictions, creating summary plots and a force plot for one fraud case.
   - I added Partial Dependence Plots (PDPs) to see how V17 and its interaction with V14 impact fraud predictions.
- What I Found:
   - Feature Importance: V17_boost (0.181), V17 (0.172), V14 (0.117), V12 (0.093), V10 (0.089) were the top features.
   - SHAP Analysis:
        - V12, V17_boost, and V17 were the biggest drivers of fraud predictions.
        - High V17 values usually mean fraud, but I was surprised to see that low V17 values can also point to fraud in some cases (e.g., in the force plot, a V17 of -13.6 added 0.23 to the fraud probability).
        - The force plot showed a 90% fraud probability, with V3 (0.46), V17 (0.23), and V17_boost (0.23) pushing it up.
   - PDPs:
        - The PDP for V17 showed a non-linear effect: as V17 increases from -1.0 to -0.5, the fraud probability slightly decreases (from ~0.0021925 to ~0.0021750). It then drops sharply around V17 = -0.5 to 0.5, staying low at ~0.0021750, before rising again past V17 = 0.5. This means higher V17 values (above 0.5) increase fraud likelihood, while values around -0.5 to 0.5 are linked to lower fraud probability.
        - The 2D PDP for V17 and V14 revealed that high V17 combined with low V14 values significantly increases the fraud probability, confirming their interaction as a key fraud indicator, which matches my SHAP findings.
    
#### Step 8: Getting Ready for the Real World
- What I Did:
  - I wrote a prediction function to use my model in real-time, making sure it preprocesses new transactions the same way (e.g., calculating V17_boost).
  - I tested the function on a fraud and a non-fraud transaction.
  - I thought about what I’d need to do to deploy this model in a real system.
- What I Found:
  - My function worked well:
       - Fraud transaction: 90% probability of fraud.
       - Non-fraud transaction: 0% probability of fraud.
  - I came up with these deployment ideas:
       - Use a Flask API for real-time predictions.
       - Make sure preprocessing matches my training setup.
       - Retrain the model regularly to catch new fraud patterns.
       - Monitor false positives and negatives in production.
       - Use AWS Lambda to handle lots of transactions.
   
#### 📈 Results
##### How My Model Performed
- Best Model: Random Forest (RF).
- AUPRC: 0.880, which I think is really good for such an imbalanced dataset.
- Cost: $8,650, the lowest among all models, meaning it balances missed frauds and false alarms well.
- Threshold: 0.560, which I found to be a good balance.
- Real-Time Testing:
   - Fraud transaction: 90% probability.
   - Non-fraud transaction: 0% probability.
 
##### What I Learned About V17
- In EDA: I found V17 has a spike ratio of 13.23, so it’s a big clue for fraud.
- In Feature Engineering: I made V17_boost to focus on extreme V17 values, and it became the top feature (importance: 0.181).
- In Oversampling: I used V17_boost to pick which fraud cases to oversample, making sure my model learned the right patterns.
- In Interpretation: SHAP showed V17 as a top feature (3rd after V12 and V17_boost). High V17 values usually mean fraud, but low values can also point to fraud sometimes, which was interesting to see.

##### Top Features
- RF Feature Importance: V17_boost (0.181), V17 (0.172), V14 (0.117), V12 (0.093), V10 (0.089).
- SHAP Analysis: V12, V17_boost, V17 were the biggest drivers of fraud predictions.

#### 🔑 What I Discovered
- My Model Works Well: My RF model is really good at spotting fraud (AUPRC: 0.880, cost: $8,650). It beats the other models I tried and doesn’t miss too many frauds or raise too many false alarms.
- V17 Was Key: I was right to focus on V17. It showed up as important in every step—EDA, feature engineering, oversampling, and interpretation. I learned that high V17 values usually mean fraud, but low values can also be a sign in some cases, which was a surprise.
- New Features Helped: The features I added, like V17_boost and Time_cycle, made my model better by focusing on the patterns I found in EDA.
- Understanding My Model: Using SHAP and PDPs, I could see exactly why my model makes certain predictions, like how V17 and V14 work together to spot fraud. The PDP showed me that V17’s effect isn’t straightforward—it dips in the middle before going up, which was a cool insight.
- It’s Stable: I made sure my code gives the same results every time by setting random_state=42 and picking the same fraud case for the SHAP force plot, so I know my results are reliable.

#### 🖥️ How I Built It
##### Tools I Used
- Programming Language: Python 3.11.
- Environment: Google Colab with Google Drive to store my data.
- Libraries:
     - For data: pandas, numpy.
     - For plots: seaborn, matplotlib, scipy (gaussian_kde).
     - For machine learning: scikit-learn (e.g., train_test_split, RandomForestClassifier, PartialDependenceDisplay).
     - For interpretation: shap.

##### What’s in My Project
- Fraud_Detection.ipynb: My main notebook with all my code and results.
- Creditcard.csv: The dataset I used (you can get it from Kaggle).
- README.md: This report I wrote to explain everything.
- Requirements.txt: The libraries you need to run my code.

##### How to Run My Code
1. What You Need:
    - Python 3.11+.
    - Google Colab or Jupyter Notebook on your computer.
    - Libraries: pandas, numpy, scikit-learn, seaborn, matplotlib, shap.
2. Steps:
- Clone my repository: git clone https://github.com/yourusername/credit-card-fraud-detection.git.
- Install the libraries: pip install -r requirements.txt.
- Open Fraud_Detection.ipynb in Google Colab.
- Mount your Google Drive and put creditcard.csv in /content/drive/MyDrive/.
- Run the cells one by one to see my results.

##### What You’ll See:
- My model’s performance: AUPRC 0.880, cost $8,650.
- The SHAP force plot: 90% fraud probability for my chosen case.
- My final report in Step 8 with all my findings.

##### What I’d Do Next
- Tweak My Model: I’d try tuning the RF model to see if I can get an even better AUPRC.
- Build an API: I’d love to turn my prediction function into a real Flask API.
- Think About Ethics: I’d look into how false positives affect customers and check for any biases in my model.
- Test in Production: I’d simulate a real system to see how my model does over time.

#### 🌟 Why This Matters to Me
I’m really proud of this project because it shows what I can do with data science. I took a tough problem—detecting fraud in a super imbalanced dataset—and built a model that works well and makes sense. Focusing on V17 was my idea, and seeing it play such a big role in my results felt rewarding. I also learned a lot about interpreting models with SHAP and PDPs, which I think is so important for real-world use. My model is ready to go live, and I’ve thought about how to make it work in a real system, which makes me excited about using data science to solve real problems.

