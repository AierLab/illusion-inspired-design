from generator import *
from planner import Planner
from concurrent.futures import ThreadPoolExecutor
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dataset generation script')
    parser.add_argument('--train_dir', type=str, help='Path to the training directory')
    parser.add_argument('--test_dir', type=str, help='Path to the testing directory')
    args = parser.parse_args()

    planner = Planner(train_dir=args.train_dir, test_dir=args.test_dir, positive_ratio=0.5)

    datasets = [dataset01, dataset02, dataset03, dataset04, dataset05, dataset06, dataset07, 
                dataset08, dataset09, dataset10, dataset11, dataset12, dataset13, dataset14]
    
    for dataset in datasets:
        try:
            planner.generate(dataset, size=2000)
            print(f"Successfully completed generation of {dataset.__name__}")
        except Exception as e:
            print(f"Error during generation of {dataset.__name__}: {str(e)}")
            # The checkpoint_generation decorator will already have saved progress
        
if __name__ == "__main__":
    main()