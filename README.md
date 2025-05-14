To reproduce the experiments, you can simply clone this repository and install the requierements in a new virtual env as followed:

```
git clone https://github.com/ValentinLafargue/Inspection.git
cd Inspection
python3 -m venv inspection
source inspection/bin/activate (or ./inspection/Script/activate given your setup)
pip install -r requierements.txt
```

In those experiments we use the following datasets:
- Adult census dataset -> https://www.kaggle.com/datasets/uciml/adult-census-income
- Folktables : Income, Mobility, Travel Time, Employment, Public Coverage -> https://github.com/socialfoundations/folktables
- Bank Account Fraud -> https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

The github is organised this way : 
- Data : contains the datasets in csv which we used
- Pre-processing : 2 jupyter notebooks which download or/and preprocess the BAF / Folktables dataets into ready to use datasets.
- src : the folder where we keep the python function within the .py
      - distance : manly different ways to compute the Wasserstein distance or the Kullback-Leibler divergence
      - GEMS3_base_explainer : The file having the entropic projection fonction (with both balanced and proportional case)
      - mitigation_fct : Function enabling fairness unbaisedness in particular Matching_W(X,S,Y), Replace (S,Y)
      - Gems_Wasserstein : the file having the wasserstein projection and semi-discret algorithm (with balanced and proportional case, as well as with or without 1D-projection)
      - sampling : useful functions regarding sampling and attempts to optimize sampling
      - utils : regular utils file
- Project : This folder is composed of the notebooks used in the project : from the unbiasedness to the method evaluations and statistical tests.
      - The notebooks (ADULT/BAF/Folktable)_unbiasedness show how we remove the DI bias (to 0.8) from the dataset with every method, we register every result including the neural network's weight for reproductibility.
      - Result Analysis : Notebook where we evaluate the distance from the modified dataset and the original one (for W and KL ; and on (X,S,Y) and (S,Y)) as well as the 5 statistical tests. (as they can take some time to compute, we register the results within the json files dic_threshold.json and dic_test_result.json)
      - Highest_undetected_unbiasing : Notebook where we find to which extent we can increase the Disparate Impact of the original datasets without being detected by any of the 5 tests with the Matching_W(X,S,Y) method. (result in a json file as well : dic_unbiasing.json)
      - W_KL_unbiasedness_result : Notebook where we study the distance of the modified dataset with the original one with different Disparate Impact unbiasing goals.
      - Simulated_exp : Experiences done on a simulated dataset where we compare for instance our optimized method against manual unbiasing method on the Wassserstein distance or the Kullback-Leibler on (S,Y).
  - Result folder : where we stock the results
 
    The csv in Data as well as some of the Results were to heavy for the Github 50Mo limit, hence they are downloadable here :
    https://drive.google.com/drive/folders/1LoHXfnelYLZf0b8Dbo3OqqlaKo-51IPY?usp=sharing
