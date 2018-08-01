==========================================================================================================================
>>>> Insurance Charge Prediction
==========================================================================================================================

>>> Overview

The dataset given is downloaded from Kaggle. The objective of the project is to predict the premium/charge of insurance for
a new customer based on the data provided and the parameters.

==========================================================================================================================

>>> Dataset

Number Of Observations: 1338

Age	   - Age of the customer
BMI     - Body mass index
Sex     - Gender of the customer
Smoker  - Does the customer smoke or not
Region  - Region were he/she belongs

==========================================================================================================================

>>>Processing

It was found that customer who were smokers had higher charges than the customer who did not smoke.Outliers were removed 
based on this. New column obesity was created based on the fact that person with bmi>30 is considered as obesity.

==========================================================================================================================

>>>Result

The premium/charge was predicted with a rms value 1500-1700 approx. The variation was mainly due to lack of data for the
algorithm to learn.The dataset contained only 1338 observation. 

==========================================================================================================================












