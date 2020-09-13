import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("dataset", type=str, default="RAP")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--debug", action='store_false')

    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--train_epoch", type=int, default=50)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lr_ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr_new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--train_split", type=str, default="trainval", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument('--device', default=0, type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')
    
    parser.add_argument('--train_transform', type=json.loads, default='{"Order": ["Resize", "Pad", "RandomCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"], "Resize": {"size": [256, 192]}, "Pad": {"padding": 10}, "RandomCrop": {"size": [256, 192]}, "RandomHorizontalFlip": {}, "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}')

    parser.add_argument('--valid_transform', type=json.loads, default='{"Order": ["Resize", "ToTensor", "Normalize"], "Resize": {"size": [256, 192]}, "Normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}}')

    return parser
