import argparse
from utils.config_loader import load_config
from pipelines import train, test, inference


def main(args):
    config = load_config(args.config)

    if args.pipeline == 'train':
        train.train_wrapper(config)
    elif args.pipeline == 'test':
        test.test_wrapper(config)
    # elif args.pipeline == 'inference':
    #     inference.inference_wrapper(config)
    else:
        raise ValueError(f'Unsupported pipeline: {args.pipeline}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Customizable Image Retrieval Framework'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to config file'
    )
    parser.add_argument(
        '--pipeline',
        type=str,
        required=True,
        help='Pipeline to run: train, test, inference',
    )
    args = parser.parse_args()
    main(args)
