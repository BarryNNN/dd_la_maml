import ipdb

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, sizes, dataset='mnist', args=None):

        net_list = []
        if "mnist" in dataset:
            if model_type=="linear":
                for i in range(0, len(sizes) - 1):
                    net_list.append(('linear', [sizes[i+1], sizes[i]], ''))
                    if i < (len(sizes) - 2):
                        net_list.append(('relu', [True], ''))
                    if i == (len(sizes) - 2):
                        net_list.append(('rep', [], ''))
                return net_list

        elif dataset == "tinyimagenet":

            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [640, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [640, 640], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 640], '')
                ]

        elif dataset == "cifar100":


            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [320, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [320, 320], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 320], '')
                ]

            # ConvNetD3: depth=3, width=128, relu, instancenorm (using bn), avgpooling
            # Input: 32x32 -> after 3 avgpool(2,2): 4x4
            # Final feature: 128 * 4 * 4 = 2048
            elif model_type == 'convnetd3':
                channels = 128  # net_width
                return [
                    # Block 1: conv -> bn -> relu -> avgpool
                    ('conv2d', [channels, 3, 3, 3, 1, 1], ''),  # 32x32 -> 32x32
                    ('bn', [channels], ''),
                    ('relu', [True], ''),
                    ('avg_pool2d', [2, 2, 0], ''),  # 32x32 -> 16x16

                    # Block 2: conv -> bn -> relu -> avgpool
                    ('conv2d', [channels, channels, 3, 3, 1, 1], ''),  # 16x16 -> 16x16
                    ('bn', [channels], ''),
                    ('relu', [True], ''),
                    ('avg_pool2d', [2, 2, 0], ''),  # 16x16 -> 8x8

                    # Block 3: conv -> bn -> relu -> avgpool
                    ('conv2d', [channels, channels, 3, 3, 1, 1], ''),  # 8x8 -> 8x8
                    ('bn', [channels], ''),
                    ('relu', [True], ''),
                    ('avg_pool2d', [2, 2, 0], ''),  # 8x8 -> 4x4

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    # classifier: 128 * 4 * 4 = 2048 -> num_classes
                    ('linear', [sizes[-1], 2048], '')
                ]

        elif dataset == "cifar10":


            if model_type == 'pc_cnn':
                channels = 160
                return [
                    ('conv2d', [channels, 3, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('conv2d', [channels, channels, 3, 3, 2, 1], ''),
                    ('relu', [True], ''),

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    ('linear', [320, 16 * channels], ''),
                    ('relu', [True], ''),

                    ('linear', [320, 320], ''),
                    ('relu', [True], ''),
                    ('linear', [sizes[-1], 320], '')
                ]

            # ConvNetD3: depth=3, width=128, relu, instancenorm (using bn), avgpooling
            # Input: 32x32 -> after 3 avgpool(2,2): 4x4
            # Final feature: 128 * 4 * 4 = 2048
            elif model_type == 'convnetd3':
                channels = 128  # net_width
                return [
                    # Block 1: conv -> bn -> relu -> avgpool
                    ('conv2d', [channels, 3, 3, 3, 1, 1], ''),  # 32x32 -> 32x32
                    ('bn', [channels], ''),
                    ('relu', [True], ''),
                    ('avg_pool2d', [2, 2, 0], ''),  # 32x32 -> 16x16

                    # Block 2: conv -> bn -> relu -> avgpool
                    ('conv2d', [channels, channels, 3, 3, 1, 1], ''),  # 16x16 -> 16x16
                    ('bn', [channels], ''),
                    ('relu', [True], ''),
                    ('avg_pool2d', [2, 2, 0], ''),  # 16x16 -> 8x8

                    # Block 3: conv -> bn -> relu -> avgpool
                    ('conv2d', [channels, channels, 3, 3, 1, 1], ''),  # 8x8 -> 8x8
                    ('bn', [channels], ''),
                    ('relu', [True], ''),
                    ('avg_pool2d', [2, 2, 0], ''),  # 8x8 -> 4x4

                    ('flatten', [], ''),
                    ('rep', [], ''),

                    # classifier: 128 * 4 * 4 = 2048 -> num_classes
                    ('linear', [sizes[-1], 2048], '')
                ]


        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)



 