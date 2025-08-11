from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt

def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-v', '--verbose', type=bool, default=False, nargs='?', const=True)

    return parser.parse_args()

def main():
    args = get_args()

    losses = list()
    with open(args.file, 'r') as input_file:
        for line in input_file.readlines():
            line_parts = line.split()
            line_parts[0] = int(line_parts[0])
            line_parts[1] = float(line_parts[1])
            if args.verbose:
                print(f'Loss: {line_parts[1]}')
            losses.append(line_parts)
    
    plt.plot(list(map(lambda x: x[0], losses)), list(map(lambda x: x[1], losses)))
    #plt.yscale('log')
    plt.ylim(bottom=0)
    #plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    main()