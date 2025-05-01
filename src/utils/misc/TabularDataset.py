from src.utils.misc.helpers import *
from src.utils.misc.constants import *


class TabularDataset(Dataset):
    """
    Custom dataset class for loading tabular data from a CSV file.

    - self.data: Pandas DataFrame containing tabular data
    - self.features: Feature matrix (excluding target column and specified columns_to_drop)
    - self.labels: Target column (if exists)
    - self.ids: Unique identifiers (previously isic_ids)
    - self.transform: Transformation applied to features (default: convert to tensor)
    """

    class PipelineWrapper(BaseEstimator, TransformerMixin):
        """
        Wrap a sklearn.pipeline.Pipeline class object and tabular dataset, preserve pd.DataFrame during pipeline execution
        - self.pipeline: sklearn.pipeline.Pipeline class object
        - self.data: pd.DataFrame class object to preprocess
        - self.target: pd.Series class object, response of self.data, mainly used for encoder
        - self.index: self.data.index
        - self.columns: self.data.columns
        """

        def __init__(self, pipeline, data, target=None):
            self.pipeline = pipeline
            self.data = data.copy()
            if isinstance(target, pd.DataFrame):
                self.target = target.copy()
            elif isinstance(target, pd.Series):
                self.target = target.to_frame().copy()
            else:
                if target is None:
                    print(f"target of `PipelineWrapper` is None")
                else:
                    raise TypeError(
                        "target of `PipelineWrapper` must be one of pd.DataFrame/pd.Series/None !"
                    )

            if isinstance(self.data, pd.DataFrame):
                self.index = self.data.index
                self.columns = self.data.columns
            else:
                raise TypeError(
                    "Input data of `PipelineWrapper` should be `pandas.DataFrame` object!"
                )

        def fit_transform(self, X=None):
            """
            Execute pipeline and return processed tabular data
            """
            for name, transformer in self.pipeline.steps:
                print("-" * 20)
                print(f"Implement transformer {name}")
                if isinstance(transformer, ColumnTransformer):  # ColumnTransformer
                    for subname, subtransformer, features in transformer.transformers:
                        print("*" * 20)
                        print(
                            f"Implement subtransformer {subname} of ColumnTransformer"
                        )
                        if subname == "target_encoder":
                            self.data[features] = subtransformer.fit_transform(
                                self.data[features], self.target
                            )
                        else:
                            self.data[features] = subtransformer.fit_transform(
                                self.data[features]
                            )
                        print("*" * 20)
                    self.data = pd.DataFrame(
                        self.data, index=self.index, columns=self.columns
                    )
                elif name in ["scaler", "imputer"]:  # Scaler or Imputer
                    self.data = pd.DataFrame(
                        transformer.fit_transform(self.data),
                        index=self.index,
                        columns=self.columns,
                    )
                else:
                    raise TypeError(
                        f"Procedure for {name} undefined in `PipelineWrapper`!"
                    )
                print("-" * 20)

            return self.data

    def __init__(
        self,
        csv_file_path,
        pipeline,
        target_column=None,
        id_column=None,
        columns_to_drop=None,
        transform=None,
        dtype=DTYPE,
    ):
        self.data = pd.read_csv(csv_file_path, low_memory=False)  # Load CSV data
        self.ids = self.data[id_column].astype(str).tolist()  # Convert IDs to strings

        # Extract labels if target_column is provided
        if target_column:
            self.labels = torch.tensor(self.data[target_column].to_numpy(), dtype=dtype)
        else:
            self.labels = None

        # Set default transformation
        self.dtype = dtype
        self.transform = (
            transform
            if transform
            else transforms.Lambda(lambda x: torch.tensor(x, dtype=dtype))
        )

        # Define columns to drop (id_column + target_column + additional drops)
        columns_to_remove = (
            [id_column]
            + ([target_column] if target_column else [])
            + (columns_to_drop if columns_to_drop else [])
        )
        columns_to_remove = list(set(columns_to_remove))  # Remove duplicates

        # Extract features (drop specified columns)
        self.features = self.data.drop(
            columns=columns_to_remove
        )  # Remove unused features
        self.pipeline = self.PipelineWrapper(
            pipeline, self.features, self.data[target_column]
        )  # Wrap pipeline
        # Apply preprocessing pipeline
        self.features = self.pipeline.fit_transform(None)
        # Ensure all features are numerical and convert to numpy.array
        self.features = self.features.apply(pd.to_numeric, errors="coerce").to_numpy()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        x_tabular = self.transform(self.features[idx])  # Apply transformation

        if self.labels is not None:
            y = self.labels[idx]
            return sample_id, x_tabular, y
        return sample_id, x_tabular, None
