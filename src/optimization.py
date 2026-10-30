import logging

from pathlib import Path
from typing import List

from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

# Local imports from: https://github.com/Alejandro-ZZ/machine-learning
from ml_engine.optuna.base_optimizer import OptunaParamSpec
from ml_engine.optuna.sklearn_optimizer import SklearnOptunaOptimizer
from ml_engine.datasets.base_dataset import BaseDataset


logger = logging.getLogger("Optimization")


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


def get_dataset() -> BaseDataset:
    """Load the dataset for optimization."""
    pass


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
        dataset=get_dataset()
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