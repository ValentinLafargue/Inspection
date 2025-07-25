This repository includes the code used to generate the result for our paper results. 
For more information about the methods and the choice we took in their implementation, we invite you to see our paper.

# Install
To reproduce the experiments, you can simply clone this repository and install the requierements in a new virtual env as followed:

```
git clone ValentinLafargue/Inspection
cd Inspection
python3 -m venv inspection
source inspection/bin/activate (or ./inspection/Script/activate given your setup)
pip install -r requirements.txt
```

# Datasets
In those experiments we use the following datasets:
- Adult census dataset [1]
- Folktables : Income, Mobility, Travel Time, Employment, Public Coverage [2]
- Bank Account Fraud [3]
- CelebA dataset [4]

# Presentation & Organization 

The github is organised this way : 
<details>
<summary> Data </summary>
      
contains the datasets in csv which we used, the csv are naturally too heavy for the Github 50Mo limit, hence they are downloadable here: [drive](https://drive.google.com/drive/folders/1LoHXfnelYLZf0b8Dbo3OqqlaKo-51IPY)
</details>

<details>
<summary> Pre-processing </summary>
      
2 jupyter notebooks which download and / or preprocess the BAF / Folktables dataets into ready to use datasets.
</details>

<details>
<summary> src : the folder where we keep the python function within the .py </summary>
      
- distance : manly different ways to compute the Wasserstein distance or the Kullback-Leibler divergence
- GEMS3_base_explainer : The file having the entropic projection fonction (with both balanced and proportional case). Original idea from [5].
- mitigation_fct : Function enabling fair-washing, in particular Matching_W(X,S,Ŷ), Replace (S,Ŷ)
- Gems_Wasserstein : the file having the wasserstein projection and semi-discret algorithm (with balanced and proportional case, as well as with or without 1D-projection)
- sampling : useful functions regarding sampling and attempts to optimize sampling
- utils : regular utils file
</details>

<details>
<summary> Project : This folder is composed of the notebooks used in the project : From the training and prediction with torch models along with the fairness evaluation to the fair-washing methods studies and the fraud detection by statistical tests. </summary>
      
- The notebooks (ADULT/BAF/Folktable)_fairwashing show how we remove the DI bias (to 0.8) from the dataset with every method, we register every result including the neural network's weight for reproductibility.
- Result Analysis : Notebook where we evaluate the distance from the modified dataset and the original one (for W and KL ; and on (X,S,Ŷ) and (S,Ŷ)) as well as the 5 statistical tests. (as they can take some time to compute, we register the results within the json files dic_threshold.json and dic_test_result.json)
- Highest_undetected_fairwashing : Notebook where we find to which extent we can increase the Disparate Impact of the original datasets without being detected by any of the 5 tests with the Matching_W(X,S,Ŷ) method. (result in a json file as well : dic_unbiasing.json)
- W_KL_fairwashing_result : Notebook where we study the distance of the modified dataset with the original one with different Disparate Impact fairwashing goals.
- Simulated_exp : Experiences done on a simulated dataset where we compare for instance our optimized method against manual fair-washing method on the Wassserstein distance or the Kullback-Leibler on (S,Ŷ). 
- Sampling_size_analysis : Short notebook highlighting the sample size impact on the Adult dataset.
- CelebA_exp : Notebook where we study the usefulness of our test on non-tabular data, on the CelebA dataset: in particular, we study using statistical tests based on the last hidden layer of CNN classifiers.
</details>

<details>
<summary> Result folder : where we stock the results </summary>
      
Some of the Results were too heavy for the Github 50Mo limit, hence they are downloadable here: [drive](https://drive.google.com/drive/folders/1LoHXfnelYLZf0b8Dbo3OqqlaKo-51IPY)
</details>


# Citing illusion of fairness : 

```


```

# References

```
[1]: Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository. DOI:305
https://doi.org/10.24432/C5XW20.306, https://www.kaggle.com/datasets/uciml/adult-census-income.

[2]: Ding, F., Hardt, M., Miller, J., and Schmidt, L. (2021). Retiring adult: New datasets for fair machine311
learning. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W., editors, Advances in312
Neural Information Processing Systems.313, https://github.com/socialfoundations/folktables.

[3]: Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R. P., Gama, J., and Bizarro, P. (2022).317
Turning the tables: Biased, imbalanced, dynamic tabular datasets for ml evaluation. In Advances318
in Neural Information Processing Systems, https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022.

[4]: Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou, December 2015, 
Deep Learning Face Attributes in the Wild, in Proceedings of International Conference on Computer Vision (ICCV)
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html or https://www.kaggle.com/datasets/jessicali9530/celeba-dataset

[5]: François Bachoc, Fabrice Gamboa, Max Halford, Jean-Michel Loubes, Laurent Risser, (2023)
Explaining machine learning models using entropic variable projection
in Information and Inference: A Journal of the IMA, Volume 12, Pages 1686–1715, https://doi.org/10.1093/imaiai/iaad010
```
