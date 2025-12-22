import pandas as pd

df = pd.DataFrame({"Couleur": ["Rouge", "Bleu", "Vert", "Rouge"]})
df_dummies = pd.get_dummies(df, columns=["Couleur"], drop_first=False)

print(df_dummies)