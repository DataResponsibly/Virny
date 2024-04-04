# BankMarketingDataset

Dataset class for the Bank Marketing dataset that contains sensitive attributes among feature columns. Source: https://github.com/tailequy/fairness_dataset/blob/main/experiments/data/bank-full.csv General description and analysis: https://arxiv.org/pdf/2110.00530.pdf (Section 3.1.5) Broad description: https://archive.ics.uci.edu/dataset/222/bank+marketing



## Parameters

- **subsample_size** (*int*) – defaults to `None`

    Subsample size to create based on the input dataset

- **subsample_seed** (*int*) – defaults to `None`

    Seed for sampling using the sample() method from pandas




