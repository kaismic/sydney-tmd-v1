# Sydney TMD V1

The main goal of this project is to develop the best performing model (macro f1) trained using only Sydney dataset. The plan is to use this model in a real-world transport mode inference mobile application.

## Implementation
- Copy a majority of the implementation details from sydney-window-size-comparison. However, the major difference is that this project does not use NOR-TMD dataset at all.
- Only include sessions with "collection_country_code" == "AU"
- Primary evaluation metric is macro f1
- Use Leave-one-out cross validation
- 3 random seeds (seeds 42, 43, and 44)
- 5000 bootstrap with 95% confidence interval chart
- Use window length 100, 120, and 140 seconds. overlap size is half window length.
- Features
    - Accelerometer and gyroscope: mean, minimum, range, standard deviation
    - Magnetometer: minimum, range, standard deviation
- This experiment will be run locally so do not configure AWS.
- Use XGBoost algorithm
- No maximum session duration