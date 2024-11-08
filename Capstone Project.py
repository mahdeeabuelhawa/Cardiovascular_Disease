#!/usr/bin/env python
# coding: utf-8

# # IMPORTS

# In[1]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier


# In[2]:


df = pd.read_csv('CVD_Capstone.csv')


# # EDA

# In[3]:


df.head(11)


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


df.drop_duplicates()


# In[10]:


df.isnull()


# In[11]:


df.dropna()


# In[12]:


df.dropna(axis=1)


# In[13]:


df.duplicated()


# In[19]:


df = df.rename(columns={'id':'ID',
                        'age':'AGE',
                        'gender': 'Gender',
                        'height':'Height',
                        'weight': 'Weight',
                        'ap_hi': 'SYS',
                        'ap_lo':'DI',
                        'cholesterol':'Cholesterol',
                        'gluc':'Glucose',
                        'smoke':'Smoke',
                        'alco':'Alcohol',
                        'active':'Active',
                        'cardio':'CVD'})


# In[20]:


print(df.head())


# In[21]:


mean_value = df.mean()

print("Mean:", mean_value)


# # Train-Test-Split

# In[22]:


X = df.drop(columns =['CVD'])
y = df['CVD']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[23]:


X_test.shape


# In[24]:


X_train.shape


# In[25]:


y.shape


# # Heat Map

# In[26]:


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5)
plt.title('Heatmap of Correlation Matrix for CVD Dataset')
plt.show()


# # Logistic Regression Model

# In[27]:


model1 =LogisticRegression()


# In[28]:


model1.fit(X_train,y_train)


# In[29]:


y_pred = model1.predict(X_test)


# In[30]:


model1.score(X_test,y_test)


# In[31]:


conf_matrix=(confusion_matrix(y_test,y_pred))


# In[32]:


print(classification_report(y_test,y_pred))


# In[33]:


sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[248]:


coefficients = model1.coef_[0]  


feature_importance1 = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
feature_importance1 = feature_importance1.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance1, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from Logistic Regression')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# # Random Forest

# In[35]:


model2 = RandomForestClassifier()


# In[36]:


model2.fit(X_train, y_train)


# In[37]:


y_pred2 = model2.predict(X_test)


# In[38]:


model2.score(X_test,y_test)


# In[39]:


conf_matrix2=(confusion_matrix(y_test,y_pred2))


# In[40]:


print(classification_report(y_test,y_pred2))


# In[41]:


sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[42]:


importances2 = model2.feature_importances_


feature_importance2 = pd.DataFrame({'Feature': X.columns, 'Importance': importances2})
feature_importance2 = feature_importance2.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance2, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# # DecisionTree

# In[43]:


model3 =DecisionTreeClassifier()


# In[44]:


model3.fit(X_train,y_train)


# In[45]:


y_pred3 = model3.predict(X_test)


# In[46]:


model3.score(X_test,y_test)


# In[47]:


conf_matrix3=(confusion_matrix(y_test,y_pred3))


# In[48]:


print(classification_report(y_test,y_pred3))


# In[49]:


sns.heatmap(conf_matrix3, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[244]:


importances3 = model3.feature_importances_


feature_importance3 = pd.DataFrame({'Feature': X.columns, 'Importance': importances3})
feature_importance3 = feature_importance3.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance3, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from DecisionTree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# # GradientBoostingClassifier

# In[51]:


model4 = GradientBoostingClassifier()


# In[52]:


model4.fit(X_train,y_train)


# In[53]:


y_pred4=model4.predict(X_test)


# In[54]:


model4.score(X_test,y_test)


# In[55]:


conf_matrix4=(confusion_matrix(y_test,y_pred4))


# In[56]:


sns.heatmap(conf_matrix4, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[245]:


importances4 = model4.feature_importances_


feature_importance4 = pd.DataFrame({'Feature': X.columns, 'Importance': importances4})
feature_importance4 = feature_importance4.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance4, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance from GradientBoostingClassifier')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# # Stacking

# In[75]:


estimators = [
    ('dtc',model3),
    ('rfc',model2),
    ('gbc',model4)
]


# In[76]:


sc = StackingClassifier(
    estimators=estimators,
    final_estimator = model1
)


# In[77]:


sc.fit(X_train,y_train)


# In[73]:


y_pred5=sc.predict(X_test)


# In[74]:


sc.score(X_test,y_test)


# In[222]:


conf_matrix5=(confusion_matrix(y_test,y_pred5))


# In[223]:


sns.heatmap(conf_matrix5, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# # Smoking

# In[188]:


df.Smoke.value_counts()


# In[191]:


df.Smoke.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["blue" , "red"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[238]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='Smoke', hue='CVD', palette=["blue" ,"red"])
axis.bar_label(axis.containers[0]);
axis.bar_label(axis.containers[1]);

plt.title('Heart Disease by smoke')
plt.xlabel('smoke ')
plt.ylabel('Count')
plt.show()


# In[243]:


count_data = df.groupby(['Smoke', 'CVD']).size().reset_index(name='Count')


total_counts = count_data.groupby('Smoke')['Count'].sum().reset_index(name='Total')


count_data = count_data.merge(total_counts, on='Smoke')

count_data['Percentage'] = (count_data['Count'] / count_data['Total']) * 100

table_data = count_data.pivot(index='Smoke', columns='CVD', values='Percentage').fillna(0)


table_data.reset_index(inplace=True)

print(table_data)


# In[219]:


df.groupby("Gender")["Smoke"].value_counts()


# In[198]:


sns.countplot(x="Gender", hue="Smoke", data=df, palette="Set1")

plt.title("Distribution of Smoking Status by Gender")
plt.xlabel("Gender Women =1 Men = 2")
plt.ylabel("Count")

plt.show()


# # Alcohol

# In[201]:


df.Alcohol.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["blue" , "red"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[203]:


df[["Alcohol" , "Smoke"]].corr()


# In[204]:


df.groupby("Smoke")["Alcohol"].value_counts()


# In[205]:


sns.barplot(x="Smoke", y="count", hue="Alcohol", data=df.groupby(["Smoke", "Alcohol"]).size().reset_index(name='count'), palette="pastel")

plt.title("Distribution of Alcohol Consumption by Smoking Status")
plt.xlabel("Smoking Status")
plt.ylabel("Count");


# In[ ]:





# # Cholesterol

# In[234]:


df.value_counts(['Cholesterol'])


# In[116]:


round( pd.crosstab(df['Cholesterol'], df['CVD'], normalize='index') * 100 , 2)


# In[119]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='Cholesterol', hue='CVD', palette=["Blue" ,"Red"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])

plt.title('Heart Disease by Cholesterol Levels')
plt.xlabel('Cholesterol Levels')
plt.ylabel('Count')
plt.show()


# # Gender

# In[228]:


cvd_counts = df[df['CVD'] == 1]['Gender'].value_counts()

plt.figure(figsize=(8, 6))
cvd_counts.index = ['Female' if x == 1 else 'Male' for x in cvd_counts.index]  
cvd_counts.plot(kind='bar', color=['#66b3ff', '#ff9999'])
plt.title('Count of Cardiovascular Disease Cases by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of CVD Cases')
plt.xticks(rotation=0)
plt.show()


# In[229]:


num_females = df[df['Gender'] == 1].shape[0]
print(f"Number of females in the dataset: {num_females}")
num_males = df[df['Gender'] == 2].shape[0]
print(f"Number of males in the dataset: {num_males}")


# In[231]:


total_males = df[df['Gender'] == 2].shape[0]  
total_females = df[df['Gender'] == 1].shape[0]  


males_with_cvd = df[(df['Gender'] == 2) & (df['CVD'] == 1)].shape[0]
females_with_cvd = df[(df['Gender'] == 1) & (df['CVD'] == 1)].shape[0]

percentage_males_cvd = (males_with_cvd / total_males * 100) if total_males > 0 else 0
percentage_females_cvd = (females_with_cvd / total_females * 100) if total_females > 0 else 0

print(f"Percentage of males with CVD: {percentage_males_cvd:.2f}%")
print(f"Percentage of females with CVD: {percentage_females_cvd:.2f}%")


# In[232]:


plt.figure(figsize=(6,5))
axis=sns.countplot(data=df, x='Gender', hue='CVD', palette='Set1' , alpha=0.8)
axis.bar_label(axis.containers[0]);
axis.bar_label(axis.containers[1]);
plt.title('Heart Disease by Gender')
plt.xlabel('Gender 1 = Males 2 =Females')
plt.ylabel('Count')
plt.show()


# # Blood Pressure

# In[151]:


df.SYS.describe()


# In[166]:


df = df[df['SYS'] >= 25]
print(df)


# In[167]:


df.SYS.describe()


# In[154]:


df.DI.describe()


# In[158]:


df[df.DI < 0]


# In[171]:


df = df[df['DI'] >= 0]
print(df)


# In[169]:


df.DI.describe()


# In[174]:


df=df[ (df.DI <=190)  & (df.SYS > 25)& (df.DI <= 240) ]
df=df[ (df.SYS <=190)  & (df.DI > 25)& (df.SYS <= 240) ]


# In[175]:


df


# In[177]:


plt.figure(figsize=(8,6)) 
sns.scatterplot(x="SYS", y="DI", data=df,  alpha=0.6)

plt.title("Scatter Plot of Systolic  vs Diastolic Blood Pressure")
plt.xlabel("Systolic Blood Pressure ")
plt.ylabel("Diastolic Blood Pressure ")


# In[178]:


correlation = df[['SYS', 'DI']].corr().iloc[0, 1]
print(f"The correlation coefficient between ap_hi and ap_lo is: {correlation:.2f}")


# In[180]:


def categorize_blood_pressure(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return 'Normal'
    elif 120 <= ap_hi < 140 or 80 <= ap_lo < 90:
        return 'Prehypertension'
    elif 140 <= ap_hi < 160 or 90 <= ap_lo < 100:
        return 'Hypertension Stage 1'
    elif 160 <= ap_hi or ap_lo >= 100:
        return 'Hypertension Stage 2'
    elif ap_hi >= 180 or ap_lo >= 120:
        return 'Hypertensive Crisis'
        
df['blood_pressure_category'] = df.apply(lambda row: categorize_blood_pressure(row['SYS'], row['DI']), axis=1)
df[['SYS', 'DI', 'blood_pressure_category']].head(10)


# In[182]:


round( pd.crosstab(df['blood_pressure_category'], df['CVD'], normalize='index') * 100 , 2)


# In[185]:


plt.figure(figsize=(10,6))
axis=sns.countplot(data=df, x='blood_pressure_category', hue='CVD', palette=["Blue" ,"red"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])
plt.title('Heart Disease by blood_pressure ')
plt.xlabel('blood_pressure')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# # GLUCOSE

# In[237]:


df['Glucose'].value_counts()


# In[124]:


glucose_1 = df[df['Glucose'] == 1]
num_cvd = glucose_1['CVD'].sum()  
total_glucose_1 = len(glucose_1) 

if total_glucose_1 > 0:
    percentage_cvd = (num_cvd / total_glucose_1) * 100
else:
    percentage_cvd = 0  

print(f"Percentage of people with glucose = 1 who have CVD: {percentage_cvd:.2f}%")


# In[127]:


glucose_2 = df[df['Glucose'] == 2]
num_cvd = glucose_2['CVD'].sum()  
total_glucose_2 = len(glucose_2) 

if total_glucose_2 > 0:
    percentage_cvd = (num_cvd / total_glucose_2) * 100
else:
    percentage_cvd = 0  

print(f"Percentage of people with above normal Glucose levels who have CVD: {percentage_cvd:.2f}%")


# In[128]:


glucose_3 = df[df['Glucose'] == 3]
num_cvd = glucose_3['CVD'].sum()  
total_glucose_3 = len(glucose_3) 

if total_glucose_2 > 0:
    percentage_cvd = (num_cvd / total_glucose_3) * 100
else:
    percentage_cvd = 0  

print(f"Percentage of people with way above normal Glucose levels who have CVD: {percentage_cvd:.2f}%")


# In[122]:


round( pd.crosstab(df['Glucose'], df['CVD'], normalize='index') * 100 , 2)


# In[142]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='Glucose', hue='CVD', palette=["blue" ,"red"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])
plt.title('Heart Disease by gluc')


# # AGE

# In[131]:


df['age_years'] = (df['AGE'] / 365).round().astype(int)


# In[132]:


df['age_years']


# In[133]:


df['age_years'].describe()


# In[139]:


df['age_years'].hist(edgecolor="white",grid=False, color="blue")


# In[135]:


df[df['age_years'] == df['age_years'].max()]


# In[136]:


df[df['age_years'] ==df['age_years'].min()]


# In[138]:


plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age_years', hue='CVD', multiple='stack', kde=False, palette='Set1')
plt.title('Heart Disease by Age')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()


# In[148]:


age_cvd_counts = df.groupby(['age_years', 'CVD']).size().unstack(fill_value=0)
age_cvd_percentages = age_cvd_counts.div(age_cvd_counts.sum(axis=1), axis=0) * 100
age_cvd_percentages.columns = ['No CVD (%)', 'CVD (%)']
age_cvd_percentages = age_cvd_percentages.reset_index()
print(age_cvd_percentages)


# # BMI

# In[80]:


df['Bmi'] = round( df['Weight'] / ((df['Height'] / 100) ** 2) , 2)


# In[81]:


df.Bmi


# In[82]:


df.Bmi.describe()


# In[84]:


df[['Bmi', 'Weight', 'Height']].corr()


# In[86]:


corr = df[['Bmi', 'Weight', 'Height']].corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(8, 5))
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        if not mask[i, j]:
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.show()


# In[89]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='Weight', y='Bmi', data=df)
plt.title('Bmi vs Weight')
plt.xlabel('Weight')
plt.ylabel('Bmi')
plt.subplot(1, 2, 2)
sns.scatterplot(x='Height', y='Bmi', data=df)
plt.title('Bmi vs Height')
plt.xlabel('Height')
plt.ylabel('Bmi')


plt.tight_layout()
plt.show()


# In[220]:


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    elif 30 <= bmi < 35:
        return 'Obesity class 1'
    elif 35<= bmi < 40:
        return 'Obesity class 2'
    else: 
        return 'Extreme Obesity'

df['BMI_category'] = df['Bmi'].apply(categorize_bmi)


# In[221]:


df['BMI_category']


# In[222]:


df['BMI_category'].unique()


# In[223]:


import matplotlib.pyplot as plt


custom_colors = ['blue', 'red', 'green', 'purple', 'pink', 'yellow'] 

df.BMI_category.value_counts().plot(
    kind="pie",
    autopct='%1.1f%%',
    colors=custom_colors,  
    explode=(0, 0, 0, 0, 0, 0),
    wedgeprops=dict(width=0.7)
)

plt.title('BMI Category Distribution')  
plt.ylabel('')  
plt.show()


# In[224]:


round( pd.crosstab(df['BMI_category'], df['CVD'], normalize='index') * 100 , 2)


# In[225]:


plt.figure(figsize=(10,6))
axis=sns.countplot(data=df, x='BMI_category', hue='CVD', palette=["blue" ,"red"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])
plt.title('Heart Disease by BMI Categories')
plt.xlabel('BMI Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# # Activity

# In[211]:


df.Active.value_counts()


# In[214]:


df.Active.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["Blue" , "red"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[218]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='Active', hue='CVD', palette=["blue", "red"])
axis.bar_label(axis.containers[0]);
axis.bar_label(axis.containers[1]);

plt.title('Heart Disease by active')
plt.xlabel('0 = Inactive 1 = Active')
plt.ylabel('Count')
plt.show()


# In[217]:


active_cvd_counts = df.groupby(['Active', 'CVD']).size().unstack(fill_value=0)


active_cvd_percentages = active_cvd_counts.div(active_cvd_counts.sum(axis=1), axis=0) * 100


active_cvd_percentages.columns = ['No CVD (%)', 'CVD (%)']


active_cvd_percentages = active_cvd_percentages.reset_index()

print(active_cvd_percentages)



# In[ ]:




