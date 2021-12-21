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
            loss = float(line.split()[-1])
            if args.verbose:
                print(f'Loss: {loss}')
            losses.append(loss)
    
    plt.plot([ i for i in range(len(losses))], losses)
    #plt.yscale('log')
    plt.ylim(bottom=0)
    #plt.xscale('log')
    plt.show()

if __name__ == '__main__':
    main()