import pandas as pd
df = pd.read_csv("classification_data_with_preds.csv")
df[["이탈여부", "예측이탈여부"]].to_csv("preds_for_confusion.csv", index=False)
