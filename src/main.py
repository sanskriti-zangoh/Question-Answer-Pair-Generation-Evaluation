import argparse

DATA_PATH = "/src/data"
RESULT_PATH = "/src/result"

def main(method, generic, collection_name, filename, n):
    print(f"Method: {method}")
    print(f"Generic: {generic}")
    print(f"Collection Name: {collection_name}")
    print(f"Filename: {filename}")
    print(f"N: {n}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument(
        "--method",
        type=str,
        choices=["separate", "combined"],
        default="combined",
        help="Method to generate Q&A pairs, either 'separate' or 'combined'. Default is 'combined'."
    )
    parser.add_argument(
        "--generic",
        action='store_true',
        help="Flag to indicate if the Q&A generation should be generic."
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="farming_market",
        help="Name of the collection in the case of 'separate' method. Default is 'farming_market'."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="farming_market.txt",
        help="Name of the file to be used as the raw data. Default is 'farming_market.txt'."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of questions per chunk. Default is 5."
    )
    parser.add_argument(
        "--t",
        type=int,
        default=20,
        help="An integer value. Default is 20."
    )

    args = parser.parse_args()
    
    main(args.method, args.generic, args.collection_name, args.filename, args.n)
