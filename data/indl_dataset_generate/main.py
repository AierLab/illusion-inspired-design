from generator import *
from planner import Planner
from concurrent.futures import ThreadPoolExecutor
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dataset generation script')
    parser.add_argument('--train_dir', type=str, help='Path to the training directory')
    parser.add_argument('--test_dir', type=str, help='Path to the testing directory')
    args = parser.parse_args()

    planner = Planner(train_dir=args.train_dir, test_dir=args.test_dir, positive_ratio=0.5)
    
    def generate_dataset(dataset):
        planner.generate(dataset, size=20000)

    datasets = [dataset02, dataset05] # dataset01, , dataset03, dataset04, 

    with ThreadPoolExecutor() as executor:
        executor.map(generate_dataset, datasets)

if __name__ == "__main__":
    main()