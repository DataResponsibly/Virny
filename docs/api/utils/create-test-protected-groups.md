# create_test_protected_groups

Create protected groups based on a test feature set.

Return a dictionary where keys are subgroup names, and values are X_test rows correspondent to this subgroup.

## Parameters

- **X_test** (*pandas.core.frame.DataFrame*)

    Test feature set

- **full_df** (*pandas.core.frame.DataFrame*)

    Full dataset

- **sensitive_attributes_dct** (*dict*)

    A dictionary where keys are sensitive attribute names (including attributes intersections),  and values are privilege values for these attributes




