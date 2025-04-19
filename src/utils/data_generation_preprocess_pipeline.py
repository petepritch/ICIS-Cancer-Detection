from helper import *
from constants import *

# Load meta data
meta_df = pd.read_csv(META_PATH)

# Drop certain features 
meta_df.drop(FEATURES_TO_DROP, axis = 1, inplace = True)

# Record index and columns
index = meta_df.index
columns = meta_df.columns

# Impute nan values
meta_df[FEATURES_TO_IMPUTE] = SimpleImputer(strategy = "most_frequent").fit_transform(meta_df[FEATURES_TO_IMPUTE])

# Target encoding location of tumor
meta_df[FEATURES_TO_ENCODE] = (
  TargetEncoder(target_type = "binary", random_state = SEED)
  .fit_transform(meta_df[FEATURES_TO_ENCODE], meta_df[RESPONSE])
)

# Enforce `age_approx` feature to numerical value (tricky and have no idea why it becomes object type)
meta_df["age_approx"] = pd.to_numeric(meta_df["age_approx"], errors = "coerce")

# Filter malignant meta data
malignant_meta_df = meta_df[meta_df[RESPONSE] == 1]
malignant_ids = malignant_meta_df.isic_id.values

# Fiter benign meta data
benign_meta_df = meta_df[meta_df[RESPONSE] == 0]
benign_ids = benign_meta_df.isic_id.values

# Store all unique ids
ids = meta_df.isic_id.values