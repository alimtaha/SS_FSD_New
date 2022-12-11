from configs.base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self):
        parser = BaseOptions.initialize(self)

        # experiment configs
        parser.add_argument('--epochs', type=int, default=25)
        parser.add_argument('--lr', type=float, default=1e-4)

        parser.add_argument('--crop_h', type=int, default=704)
        parser.add_argument('--crop_w', type=int, default=352)
        parser.add_argument('--log_dir', type=str, default='./logs')

        # logging options
        parser.add_argument('--val_freq', type=int, default=1)
        parser.add_argument('--save_freq', type=int, default=10)
        parser.add_argument('--save_model', action='store_true')
        parser.add_argument('--save_result', action='store_true')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--logarithmic', action='store_true')
        parser.add_argument('--ckpt_dir', type=str,
                            default='./ckpt/best_model_nyu.ckpt',
                            help='load ckpt path')
        parser.add_argument('--disparity', action='store_true')

        return parser
