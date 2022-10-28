import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Square function')
    parser.add_argument('--input', type=int,
                        help='input')
    parser.add_argument('--input3', type=str,
                        help='input3')
    parser.set_defaults(input = 2, input2 = 'marcelo')
    return parser.parse_args()

def main():
    args = arg_parse()
    x = args.input
    y = args.input2
    print(f"The {y} square is {x**2}")

if __name__ == '__main__':
    main()

