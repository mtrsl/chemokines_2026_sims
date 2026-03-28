import polars as pl
from itertools import product
import argparse


def expand_params(base_df: pl.DataFrame, varying: dict) -> pl.DataFrame:
    vary_df = pl.DataFrame(
        list(product(*varying.values())), schema=list(varying.keys()), orient="row"
    )
    return base_df.join(vary_df, how="cross")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("fixed_params_file", type=str)
    parser.add_argument("all_params_file", type=str)

    args = parser.parse_args()

    fixed_params_file = args.fixed_params_file
    all_params_file = args.all_params_file

    # hardcode for now. maybe parse args instead - how to pass lists of params cleanly?
    varying = {
        "chi": [0.04, 0.4, 4.0, 40.0],
        "alpha": [0.0008, 0.0065, 0.05, 0.388, 3],
        "Pe": [-5.0, 0.0, 5.0],
        "n_cells": [20, 50, 120, 210],
        "CCL21_added": ["true", "false"],
    }

    fixed_params = pl.read_csv(fixed_params_file)
    print(fixed_params)

    all_param_sets = expand_params(fixed_params, varying)
    print(all_param_sets)

    all_param_sets.write_csv(all_params_file)


if __name__ == "__main__":
    main()
