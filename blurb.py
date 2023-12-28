import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
  
# metadata 
print(rice_cammeo_and_osmancik.metadata) 
  
# variable information 
print(rice_cammeo_and_osmancik.variables) 



# Zapis do pliku Excel
data_combined = pd.concat([X, y], axis=1)  # Łączenie cech i etykiet

excel_file_path = 'rice.xlsx'  # Nazwa pliku Excel

# Zapis do pliku Excel
data_combined.to_excel(excel_file_path, index=False)