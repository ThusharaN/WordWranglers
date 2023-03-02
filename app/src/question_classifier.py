import argparse

from training import train
from testing import test

if __name__ == "__main__":
    print("Starting the word wrangler....")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Training mode - model issaved"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Testing mode - needs a model to load",
    )
    args = parser.parse_args()
    if args.train:
        train(args.config)
    elif args.test:
        test(args.config)
