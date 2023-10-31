import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler


path = r'C:\Users\marti\Desktop\MARTÍN\DATA SCIENCE\Platzi\ML_projects\logistic_regression\multinomial\Dry_Bean.csv'
df = pd.read_csv(path)

df.head(5)
df['Class'].unique()
df.describe()

df.drop_duplicates(inplace=True)
df.isnull().sum()

# Under-sampling
color_palette = ['red', 'blue', 'yellow', 'purple', 'cyan', 'orange', 'gold']
sns.countplot(x=df.Class, palette=color_palette)
plt.show()

undersample = RandomUnderSampler(random_state=42)

X = df.drop('Class', axis=1)
y = df.Class

X_over, y_over = undersample.fit_resample(X, y)

sns.countplot(x=y_over, palette=color_palette)
plt.show()

print(df.shape)
print(X_over.shape)

# Convert to numerics
print(list(np.unique(y_over)))

y_over.replace(['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA'], [1, 2, 3, 4, 5, 6, 7], inplace=True)

print(list(np.unique(y_over)))

# EDA
df_dea = X_over
df_dea['Class'] = y_over

plt.figure(figsize=(15, 10))
sns.heatmap(df_dea.corr(), annot=True)
plt.show()

X_over.drop(['ConvexArea', 'EquivDiameter'], axis=1, inplace=True)

# Data visualization
plt.figure(figsize=(5, 5))
sns.pairplot(df_dea, hue='Class')
plt.show()

# Split and Scaling
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, random_state=42, shuffle=True, test_size=0.2)
st_x = StandardScaler()
X_train = st_x.fit_transform(X_train)
X_test = st_x.fit_transform(X_test)


# Multiclass Model
def logistic_model(c_, solver_, multiclass_):
    logistic_regression_model = LogisticRegression(random_state=42,
                                                   solver=solver_,
                                                   multi_class=multiclass_,
                                                   n_jobs=-1,
                                                   C=c_)
    return logistic_regression_model


model = logistic_model(1, 'saga', 'multinomial')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

multiclass = ['ovr', 'multinomial']
solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
scores = []
params = []

for i in multiclass:
    for j in solver_list:
        try:
            model = logistic_model(1, i, j)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            params.append(i + '-' + j)
            accuracy = accuracy_score(y_test, predictions)
            scores.append(accuracy)
        except:
            None

# Results Evaluation
fig = plt.figure(figsize=(10, 10))
sns.barplot(x=params, y=scores).set_title('Beans Accuracy')
plt.xticks(rotation=90)
plt.show()

model = logistic_model(1, 'newton-cg', 'multinomial')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='gray')
plt.show()
