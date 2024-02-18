# Data classification
Data link: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik

This work presents an attempt to analyse and classify observations from a selected Machine Learning repository.
The data processing process includes a preliminary analysis of the data in the form of calculating descriptive statistics, standardising variables and using classification models (linear regression, SVM, KNN, neural networks, random forests, decision trees, LDA, QDA) to assign observables to a specific category.
Finally, we make a summary of our results in the form of a tabular statement of the different indicators that determine the accuracy of the category matching of our observations.


The subjects of this study are two rice varieties grown in Turkey - Osmanick and Cammeo. Each species is distinguished by its own characteristics.
3810 images of rice grains of both species were taken, processed and traits inferred. Eight morphological traits were obtained and stored in the repository.

With the data available for the two rice species, we wanted to classify which belong to the $Cammeo$ species and which to $Osmancik$.


# Files
- `rice.xlsx` - data from the Machine Learning repository
- `rice.ipynb` - Jupyer file showing the data classification process
- `rice.html` - a generated report based on the `rice.ipynb` file
- `requirements.txt` - required libraries to compile the code