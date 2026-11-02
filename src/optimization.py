import logging
import pandas

from pathlib import Path
from typing import List, Tuple, Union

from numpy.typing import NDArray, ArrayLike
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

# Local imports from: https://github.com/Alejandro-ZZ/machine-learning
from ml_engine.optuna.base_optimizer import OptunaParamSpec
from ml_engine.optuna.sklearn_optimizer import SklearnOptunaOptimizer
from ml_engine.datasets.base_dataset import OptimizationData


logger = logging.getLogger("Optimization")


# EXPERIMENT DATASET
# ============================================
class NeighborsDataset(OptimizationData):
    """Dataset for the Pixel-wise Segmenter experiment."""
    def __init__(
            self, 
            path: Path, 
            split_col: str, 
            features: List[str],
            targets: Union[str, List[str]],
            train_value: str = "train",
            evaluate_value: str = "test"
        ):
        # Data about the dataset splits
        self.split_col = split_col
        self.train_value = train_value
        self.validate_value = evaluate_value

        # Features (x) and target (y) columns
        self.features = features
        self.targets = targets if isinstance(targets, list) else [targets]

        # Dataset loading
        self.df = self.load_dataset(path)
    

    def load_dataset(self, path: Path) -> pandas.DataFrame:
        # Load the dataset
        try:
            file_extension = path.suffix.lower()
            if file_extension == ".parquet":
                df = pandas.read_parquet(path)
            elif file_extension == ".csv":
                df = pandas.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            raise ValueError(f"Failed to load dataset from '{path}': {str(e)}")

        # Check for required columns in the dataset 
        required_values = set([self.split_col] + self.features + self.targets)
        missing_values = required_values - set(df.columns)
        if missing_values:
            raise ValueError(f"Missing required columns in dataset: {missing_values}")

        # Check for required split values in the dataset
        required_values = set([self.train_value, self.validate_value])
        missing_values = required_values - set(df[self.split_col])
        if missing_values:
            raise ValueError(f"Missing required split values in '{self.split_col}' column: {missing_values}")

        return df.set_index(self.split_col)

    def get_split_data(self, value: str) -> Tuple[NDArray, NDArray]:
        """Retrieve the dataset's features (x) and labels (y) for a given split as a tuple (x, y)."""
        if value not in self.df.index:
            raise ValueError(f"Split value '{value}' not found in dataset '{self.split_col}' column.")
        
        split_df = self.df.loc[value]
        x = split_df[self.features].to_numpy()
        y = split_df[self.targets].to_numpy().squeeze()  # Ensure ``y`` is a 1D array if it has a single column
        return x, y

    def uses_validation(self) -> bool:
        """Indicates whether the dataset provides a separate validation set."""
        return self.validate_value in self.df.index

    def train_data(self, **kwargs) -> Tuple[NDArray, NDArray]:
        return self.get_split_data(self.train_value)

    def evaluate_data(self, **kwargs) -> Tuple[NDArray, NDArray]:
        return self.get_split_data(self.validate_value)


# EXPERIMENT DATA FUNCTIONS
# ============================================
def get_model_specs() -> List[OptunaParamSpec]:
    """Random Forest Classifier hyperparameter search space for Optuna optimization."""
    return [
        # Total number of trees in the forest
        OptunaParamSpec(
            name="n_estimators",
            type="int",
            options={"low": 20, "high": 1000, "step": 20, "log": False}
        ),

        # Maximum depth of each decision tree
        #   * We transform the value of 33 to `None`, which means no limit on the depth of the tree.
        OptunaParamSpec(
            name="max_depth",
            type="int",
            options={"low": 2, "high": 33, "step": 1, "log": False},
            transform_fnc=lambda trial, value: None if (value == 33) else value
        ),

        # Number of features to consider at each split
        #   * "sqrt": square root of the number of features
        #   * "log2": base 2 logarithm of the number of features
        #   *   None: all features are considered
        OptunaParamSpec(
            name="max_features",
            type="categorical",
            options={"choices": ["sqrt", "log2", None]}
        ),

        # Minimum number of samples required to split an internal node
        # Usually 2 times the `min_samples_leaf` value
        OptunaParamSpec(
            name="min_samples_split",
            type="int",
            options={"low": 2, "high": 100, "step": 1, "log": True}
        ),

        # Minimum number of samples required to form a leaf node
        OptunaParamSpec(
            name="min_samples_leaf",
            type="int",
            options={"low": 1, "high": 50, "step": 1, "log": True}
        ),

        # OptunaParamSpec(
        #     name="none_depth",
        #     type="categorical",
        #     options={"choices": [True, False]},
        #     pass_to_model=False  # Control parameter, not passed to model
        # ),
    ]


def get_dataset(n_neighbors: int) -> OptimizationData:
    """Load the dataset for optimization."""
    dataset_path = Path("assets/datasets/PHCO") / f"neighbors_{n_neighbors}NN.parquet"
    
    return NeighborsDataset(
        path=dataset_path,
        split_col="Split",
        features=features,
        targets=["ClassId"],
        train_value="train",
        evaluate_value="test"
    )


def get_study_config() -> dict:
    """Study configuration parameters for Optuna optimization."""
    # The pruner stops unpromising trials early (not used and dummy for our purposes)
    # Hyperband paper: https://www.jmlr.org/papers/volume18/16-558/16-558.pdf
    #   *     `min_resource` -> is the minimum number of steps (epochs) the model need to start converge
    #   *     `max_resource` -> should match the maximum iteration steps (e.g., number of epochs)
    #   * `reduction_factor` -> the paper suggest a value of euler (2.718). In practice a value of 3 or 4.
    n_epochs = 50
    pruner = HyperbandPruner(
        min_resource=10,
        max_resource=n_epochs,
        reduction_factor=3,
    )

    # Method for sampling hyperparameters
    #   * `n_startup_trials` -> total trials to sample randomly before using TPE algorithm
    #   *     `multivariate` -> assumes there might be correlations between hyperparameters
    #   *            `group` -> improve optimization search when using conditional search spaces
    #   *    `constant_liar` -> more efficiency for parallel executions (n_jobs != 1), otherwise it's useless
    #   * `constraints_func` -> evaluate combinations before starting a trial (e.g., avoid memory leak)
    sampler = TPESampler(
        n_startup_trials=10,
        multivariate=True,      # experimental (v2.2.0)
        group=False,            # experimental (v2.8.0)
        constant_liar=False,    # experimental (v2.8.0), use "heartbeat_interval" in RDBStorage
        constraints_func=None   # experimental (v3.0.0)
    )

    # Study data
    estimator_type = type(estimator).__name__
    storage_url = f"sqlite:///{output_path}/optuna_studies.db"
    study_name = f"RGB_Segmenter_{n_neighbors}NN_{estimator_type}"

    return dict(
        name=study_name,
        storage_url=storage_url,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )


# OPTIMIZATION EXECUTION
# ============================================
if __name__ == "__main__":

    # ---------------------------------
    output_path: Path = Path("results/optimization")
    estimator: BaseEstimator = RandomForestClassifier(random_state=23)
    n_neighbors: int = 8
    n_trials: int = 200
    # ---------------------------------

    # Main optimizer orchestrator
    optimizer = SklearnOptunaOptimizer(
        params=get_model_specs(),
        metrics={"f1_macro": "maximize"},
        estimator=estimator,
        dataset=get_dataset(n_neighbors)
    )

    # Study setup for the optimization process
    study_config = get_study_config()
    optimizer.setup_study( **study_config )
    logger.info(
        f"Study setup completed: "
        f"name='{study_config['name']}', "
        f"storage='{study_config['storage_url']}'"
    )
    logger.info(f"Dashboard available with: 'optuna-dashboard {study_config['storage_url']}'")

    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Run the optimization
    # https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-solve-the-error-that-occurs-when-performing-parallel-optimization-with-sqlite3
    try:
        optimizer.optimize(n_trials=n_trials, timeout=None, n_jobs=1)
    except Exception as e:
        logger.error(f"Failed to optimize: {str(e)}")
        import pdb; pdb.set_trace()
        raise e

    # Save the optimizer info
    try:
        results_file = output_path / f"{n_neighbors}NN_{estimator_type}_results.parquet"
        info_file = output_path / f"{n_neighbors}NN_{estimator_type}_metadata.json"
        
        cv_results = optimize_dataframe_dtypes(df=optimizer.cv_results_)
        cv_results.to_parquet(path=results_file, index=False)
        logger.info(f"Optimization results saved to: '{results_file.as_posix()}'")

        save_json_file(info_file, optimizer.optimizer_info)
        logger.info(f"Optimizer metadata saved to: '{info_file.as_posix()}'")
    except Exception as e:
        logger.error(f"Failed to save optimization data: {str(e)}")
        import pdb; pdb.set_trace()
        raise e

    from pprint import pprint
    print("\nOptimizer info:")
    pprint(optimizer.optimizer_info, indent=2, sort_dicts=False)
    print()