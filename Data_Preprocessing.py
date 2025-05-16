import pandas as pd
from IPython.display import display


fake_information = pd.read_csv("CoAID-master/05-01-2020/NewsFakeCOVID-19.csv") 
real_information = pd.read_csv("CoAID-master/05-01-2020/NewsRealCOVID-19.csv")


print("Fake news shape:", fake_information.shape)
print("Real news shape:", real_information.shape)


fake_clean = fake_information[['title', 'content', 'fact_check_url', 'news_url']].copy()
real_clean = real_information[['title', 'content', 'fact_check_url', 'news_url']].copy()


fake_clean['label'] = 0  # Negative outcome 
real_clean['label'] = 1  # Positive outcome 

combined_information = pd.concat([fake_clean, real_clean], ignore_index=True)

combined_information.dropna(subset=['title', 'content'], inplace=True)

combined_information.reset_index(drop=True, inplace=True)


print("\nCombined Dataset Preview:")
# print(combined_information.head(1))
display(combined_information.head(1500))


combined_information.to_csv("combined_dataset.csv", index=False)