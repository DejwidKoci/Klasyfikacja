import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 



# Zapis do pliku Excel
data_combined = pd.concat([X, y], axis=1)  # Łączenie cech i etykiet

excel_file_path = 'wine.xlsx'  # Nazwa pliku Excel

# Zapis do pliku Excel
data_combined.to_excel(excel_file_path, index=False)