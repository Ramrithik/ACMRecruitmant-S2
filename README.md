# ACMRecruitmant-S2
# Progress 
## week 1:
###  Started Strivers A2Z sheet (Array - easy)
### 1. Finished "Largest element in an array"
### 2. Finished " Finding the second largest term in an array"
![Alt](work/find_largest_term_in_array.png)
![Alt](work/find_second_largest_term_in_array.png)
### 3.Finished "Check if an Array is Sorted"
![Alt](work/check_wheather_array_sorted.png)
### 4.Finished "Remove Duplicates in-place from Sorted Array"
![Alt](work/remove_duplicates_from_array.png)
### 5.Finished "rotate an array by one"
![Alt](work/rotate_array_by_one.png)
### 6.Finished "rotate an array by n terms'
![Alt](work/rotate_an_array_by_n_terms.png)
## started revising the theory topics 
## finished topics:
### 1. like handling outliers
### i. z- score method
### ii .interquartile range
### iii. capping and clipping
### 7.Finished "move zeroes to end"
![Alt](work/move_zeros.png)
### 8. finished "Linear search"
![Alt](work/linear_search.png)
### 9. Finished "union of arrays"
![Alt](work/Union_of_arrays.png)
### 10. Finished " finding the missing term"
![alt](work/finding_missing_number.png)
### 11. Finished " finding the max conseutive ones"
![Alt](work/find_max_conseutive_ones.png)
### 12. Finished "find the number that appears once"
![Alt](work/Single_number.png)
### 13.Finished "Longest Subarray with given Sum K"
![Alt](work/Longest_Subarray_with_given_Sum_K.png)
### 14.Finished "Longest_Subarray_with_sum_K _[Postives_and_Negatives]"
![Alt](work/Longest_Subarray_with_sum_T.png)

## week 2
### Started Strivers A2Z sheet (Array - Medium)
### 1.Finished " Two Sum Problem"
![Alt](work/2_sum_problem.png)
### 2.Finished "Sort an array of 0's 1's and 2's"
![Alt](work/sort_an_array.png)
### 3.Finished "Majority_Element"
![Alt](work/Majority_Element.png)
### 4.Finished "Kadane's Algorithm, maximum subarray sum"
![Alt](work/max_sub_array.png)
### 6.Finished "Stock_Buy_and_Sell"
![Alt](work/Stock_Buy_and_Sell.png)
## ML assignment 
### completed Video Tutorials:

#### • StatQuest with Josh Starmer Linear Regression

#### • Codebasics – Gradient Descent Explained
### completed coding
### • Linear Regression from Scratch (Python) - GeeksforGeeks
#### code 
![Alt](work/Linear_reg(1).png)
![Alt](work/Linear_reg(2).png)
#### output
![Alt](work/Linear_reg_output.png)
### •  Implementing Gradient Descent in Python
#### code
![Alt](work/gradient_descent(1).png)
![Alt](work/gradient_descent(2).png)
#### output
![Alt](work/gradient_descent_output.png)
### started k-means algorithm 
#### code
![Alt](work/k-means(1).png)
![Alt](work/k-means(2).png)
#### output
![Alt](work/k-means_output.png)
# Machine Learning Mini Projects
 Week 2 Implementation Task:
- [Titanic Survival Prediction (Supervised Learning)](#titanic-survival-prediction-supervised-learning)
- [Customer Segmentation (Unsupervised Learning)](#customer-segmentation-unsupervised-learning)

## Titanic Survival Prediction (Supervised Learning)

### Problem Statement
Predict whether a passenger survived the Titanic shipwreck based on features such as age, gender, and class.

###  Dataset
- Source: [Titanic Dataset on GitHub](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)

###  Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

###  Workflow
1. **Data Cleaning**:  
   - Filled missing values for Age and Embarked.
   - Dropped irrelevant columns like Cabin, Ticket, Name, and PassengerId.
2. **Encoding**:  
   - Label Encoding for Sex and Embarked.
3. **EDA**:  
   - Visualized survival distribution across genders.
4. **Model Building**:  
   - Decision Tree Classifier
   - Random Forest Classifier
5. **Evaluation**:  
   - Accuracy Score
   - Confusion Matrix

###  Results
- Random Forest outperformed Decision Tree.
- Model accuracy > 80%.

## Customer Segmentation (Unsupervised Learning)

###  Problem Statement
Group mall customers into clusters based on their annual income and spending score.

###  Dataset
- Source: [Mall Customer Segmentation Dataset](task_implementation/customer_segmentation.csv)

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

###  Workflow
1. **Data Preprocessing**:  
   - Selected Annual Income (k$) and Spending Score (1-100) for clustering.
2. **EDA**:  
   - Scatter plot for initial distribution.
3. **Model Building**:  
   - K-Means Clustering
   - Optimal clusters determined using Elbow Method (k=5).
4. **Evaluation**:  
   - Silhouette Score calculated.

###  Results
- Five distinct customer groups identified.
- Visualized customer segments clearly.

##  How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python titanic_survival_prediction.py
python customer_segmentation.py

