# Deep-Learning-Challenge

![image](https://github.com/carojasp12/Deep-Learning-Challenge/assets/152667250/fb78cc73-ba97-48b6-9bad-b0d98fd878c2)

## Overview of the Analysis

Alphabet Soup, a nonprofit foundation, is looking to improve its funding selection process by using machine learning and neural networks. The dataset provided by Alphabet Soup includes over 34,000 funded organizations. The goal of this project is to predict the success of future applicants. Utilizing features provided in the dataset, including metadata about each organization, a binary classifier will be developed to figure out the likelihood of success if an organization receives funding from Alphabet Soup. These are the futures given in the dataset:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Results

### Data Preprocessing

- IS_SUCCESSFUL is the target variable in the model.
- EIN and NAME were removed from the data because they are neither targets nor features.
- The following are the features of the model:
    APPLICATION_TYPE
    AFFILIATION
    CLASSIFICATION
    USE_CASE
    ORGANIZATION
    STATUS
    INCOME_AMT
    SPECIAL_CONSIDERATIONS
    ASK_AMT
  
- Any APPLICATION_TYPE with fewer than 500 data points was merged with other types.
- Any CLASSIFICATION with fewer than 1800 data points was merged with other types.
- We converted categorical data to numeric with pd.get_dummies.
- We standardized the features of the dataset with StandardScaler.
  
### Compiling, Training, and Evaluating the Model

- Two hidden layers were used.
- In the first hidden layer, 80 neurons were used with the ReLU activation function for simplicity and computational efficiency.
- In the second hidden layer, 30 neurons were used with the ReLU activation function. Reducing the number of neurons helps the network condense information from the previous layer.
- An output layer with 1 neuron with a sigmoid activation function was used to provide a clear probability for the binary classification task.
  ![Screenshot 2024-06-01 191551](https://github.com/carojasp12/Deep-Learning-Challenge/assets/152667250/b7cf210f-13ac-4a32-838c-ed9105ac2d11)

- We did not achieve the target model’s performance with our first attempt.
  ![Screenshot 2024-06-01 191635](https://github.com/carojasp12/Deep-Learning-Challenge/assets/152667250/e72daf36-9a39-41ee-9d3c-10456f773b24)

- To increase model performance, we did the following:
    1. We removed the SPECIAL_CONSIDERATIONS feature.
    2. Any APPLICATION_TYPE with fewer than 100 data points was merged with other types.
    3. Any CLASSIFICATION with fewer than 200 data points was merged with other types.
    4. Three and four hidden layers were used.
    5. We experimented with other activation functions such as Tanh, Swish, and ELU.

