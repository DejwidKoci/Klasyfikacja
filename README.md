# Klasyfikacja danych
Link do danych: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik

Praca przedstawia próbę analizy i klasyfikacji obserwacji z wybranego repozytorium Machine Learningu.
Proces przetwarzania danych zawiera wstępną analizę danych w postaci obliczenia statystyk opisowych, standaryzacji zmiennych oraz wykorzystanie modeli klasyfikacyjnych (regresji liniowej, SVM, KNN, sieci neuronowych, lasów losowych, drzew decyzyjnych, LDA, QDA) by przyporządkować obseracje do konkretnej kategorii.
Na koniec robimy podsumowanie naszych wyników w postaci zestawienia tabelarycznego poszczególnych wskaźników określających dokładność dopasowania kategorii naszych obserwacji.


Przedmiotem badania są dwie odmiany ryżu uprawianych w Turcji - Osmanick oraz Cammeo. Każdy z tych gatunków wyróżnia się swoimi właściwościami.
Wykonano 3810 zdjęć ziaren ryżu obu gatunków, przetworzono je i dokonano wnioskowania o cechach. Uzyskano 8 cech morfologicznych, które zapisano w repozytorium.

Mając do dyspozycji dane dotyczące dwóch gatunków ryżu, chcieliśmy dokonać klasyfikacji, które należą do gatunku $Cammeo$, a które do $Osmancik$.


# Pliki
- `rice.xlsx` - dane z repozytorium Machine Learning
- `rice.ipynb` -  plik Jupyer przedstawiający proces klasyfikacji danych
- `rice.html` - wygenerowany raport na podstawie plikut `rice.ipynb`
