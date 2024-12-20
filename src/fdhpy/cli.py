import argparse
from inspect import Parameter, signature

import numpy as np


def cli_runner(cls, additional_kwargs=None):
    """
    Run the command-line interface for a given model class.

    This function parses command-line arguments and uses them to create an instance
    of the specified model class, allowing the user to execute its methods or access
    its properties on-the-fly.

    Parameters
    ----------
    cls : type
        The model class to instantiate (e.g., MossRoss2011, PetersenEtAl2011).

    additional_kwargs : callable, optional
        A callable that adds additional arguments to the parser, specific to the model class.

    Returns
    -------
    None
    """

    # Set print options for numpy to make the output wider
    np.set_printoptions(linewidth=300)

    # Create argument parser
    parser = argparse.ArgumentParser(description=f"Run {cls.__name__} with arguments.")

    # Method is positional
    parser.add_argument(
        "method",
        type=str,
        help="Method or property to call on the class (e.g., 'displ_site').",
    )

    # Additional positional arguments for methods
    parser.add_argument(
        "positional_args",
        nargs="*",
        help="Positional arguments for the method.",
    )

    # Define flagged arguments for the model
    parser.add_argument("-m", "--magnitude", type=float, help="Earthquake moment magnitude.")
    parser.add_argument("-s", "--style", type=str, help="Style of faulting.")
    parser.add_argument(
        "--xl", type=float, help="Normalized location x/L along the rupture length."
    )
    parser.add_argument("-p", "--percentile", type=float, help="Aleatory quantile of interest.")
    parser.add_argument("-v", "--version", type=str, help="Model version.")
    parser.add_argument("--metric", type=str, help="Definition of displacement.")

    # Add additional model-specific arguments
    if additional_kwargs:
        additional_kwargs(parser)

    # Parse arguments
    args = parser.parse_args()
    kwargs = vars(args)
    method_name = kwargs.pop("method")
    positional_args = kwargs.pop("positional_args", [])

    # Instantiate the model class
    constructor_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    obj = cls(**constructor_kwargs)

    # Access methods or properties dynamically
    try:
        attribute = getattr(obj, method_name)  # Single access point
    except AttributeError:
        print(f"The method or property '{method_name}' does not exist in {cls.__name__}.")
        return

    # Handle callable methods
    if callable(attribute):
        sig = signature(attribute)
        converted_args = []

        # Dynamically convert positional arguments based on the method signature
        for i, (param_name, param) in enumerate(sig.parameters.items()):
            if i >= len(positional_args):
                break
            arg_value = positional_args[i]

            # Convert argument to the expected type
            try:
                if param.annotation != Parameter.empty:
                    converted_args.append(param.annotation(arg_value))
                else:
                    converted_args.append(arg_value)
            except ValueError:
                print(f"ERROR: Could not convert argument '{arg_value}' to {param.annotation}.")
                return

        # Pass the converted arguments to the method
        result = attribute(*converted_args)
        print(f"\nResults for {cls.__name__} ({method_name}):")
        print(result)

    # Handle non-callable properties
    else:
        print(f"\nResults for {obj.__str__()}:", end="\n")
        if method_name in ["cdf", "prob_exceed"]:
            print(f"Displacement array: {obj.displ_array}")

        if method_name == "displ_profile":
            print(f"x/L array: {attribute[0]}")
            result = attribute[1]
        print(f"{method_name}: {attribute}", end="\n\n")
