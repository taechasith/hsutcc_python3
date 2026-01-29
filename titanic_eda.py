import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

csv_path = "Titanic-Dataset.csv"
df = pd.read_csv(csv_path)

colmap = {c.lower(): c for c in df.columns}
def C(name):
    return colmap.get(name.lower(), name)

surv = C("Survived")
sex = C("Sex")
pclass = C("Pclass")
emb = C("Embarked")
age = C("Age")
fare = C("Fare")
sibsp = C("SibSp")
parch = C("Parch")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=surv)
plt.title("Survival count")
plt.xlabel("Survived (0=No, 1=Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(14, 4))

sns.barplot(data=df, x=sex, y=surv, estimator="mean", errorbar=None, ax=ax[0])
ax[0].set_title("Survival rate by gender")
ax[0].set_ylabel("Survival rate")

sns.barplot(data=df, x=pclass, y=surv, estimator="mean", errorbar=None, ax=ax[1])
ax[1].set_title("Survival rate by passenger class")
ax[1].set_ylabel("")

sns.barplot(data=df, x=emb, y=surv, estimator="mean", errorbar=None, ax=ax[2])
ax[2].set_title("Survival rate by embarkation port")
ax[2].set_ylabel("")

plt.tight_layout()
plt.show()

num_cols = [c for c in [surv, age, fare, sibsp, parch, pclass] if c in df.columns]
corr = df[num_cols].corr(numeric_only=True)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation (Survived vs numerical features)")
plt.tight_layout()
plt.show()

print("\nSurvival rate by gender:")
print(df.groupby(sex)[surv].mean().sort_values(ascending=False))

print("\nSurvival rate by passenger class:")
print(df.groupby(pclass)[surv].mean().sort_values(ascending=False))

print("\nSurvival rate by embarkation port:")
print(df.groupby(emb)[surv].mean().sort_values(ascending=False))

print("\nCorrelation with Survived:")
print(corr[surv].drop(labels=[surv]).sort_values(key=lambda s: s.abs(), ascending=False))
