import inspect
from argparse import ArgumentParser

from torch import nn

from lasaft.source_separation.conditioned.cunet.dcun_gpocm import DenseCUNet_GPoCM, DenseCUNet_GPoCM_Framework
from lasaft.source_separation.conditioned.loss_functions import get_conditional_loss
from lasaft.source_separation.sub_modules.building_blocks import TFC
from lasaft.utils import functions


class DCUN_TFC_GPoCM(DenseCUNet_GPoCM):

    def __init__(self,
                 n_fft,
                 n_blocks, input_channels, internal_channels, n_internal_layers,
                 first_conv_activation, last_activation,
                 t_down_layers, f_down_layers,
                 kernel_size_t, kernel_size_f,
                 tfc_activation,
                 control_vector_type, control_input_dim, embedding_dim, condition_to,
                 control_type, control_n_layer, pocm_type, pocm_norm
                 ):

        tfc_activation = functions.get_activation_by_name(tfc_activation)

        def mk_tfc(in_channels, internal_channels, f):
            return TFC(in_channels, n_internal_layers, internal_channels,
                       kernel_size_t, kernel_size_f, tfc_activation)

        def mk_ds(internal_channels, i, f, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in t_down_layers else (1, 2)
            ds = nn.Sequential(
                nn.Conv2d(in_channels=internal_channels, out_channels=internal_channels,
                          kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return ds, f // scale[-1]

        def mk_us(internal_channels, i, f, n, t_down_layers):
            if t_down_layers is None:
                scale = (2, 2)
            else:
                scale = (2, 2) if i in [n - 1 - s for s in t_down_layers] else (1, 2)

            us = nn.Sequential(
                nn.ConvTranspose2d(in_channels=internal_channels, out_channels=internal_channels,
                                   kernel_size=scale, stride=scale),
                nn.BatchNorm2d(internal_channels)
            )
            return us, f * scale[-1]

        super(DCUN_TFC_GPoCM, self).__init__(
            n_fft,
            input_channels, internal_channels,
            n_blocks, n_internal_layers,
            mk_tfc, mk_ds, mk_us,
            first_conv_activation, last_activation,
            t_down_layers, f_down_layers,
            # Conditional Mechanism #
            control_vector_type, control_input_dim, embedding_dim, condition_to,
            control_type, control_n_layer, pocm_type, pocm_norm
        )

class DCUN_TFC_GPoCM_Framework(DenseCUNet_GPoCM_Framework):

    def __init__(self, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 optimizer, lr, dev_mode,
                 train_loss, val_loss,
                 **kwargs):

        valid_kwargs = inspect.signature(DCUN_TFC_GPoCM.__init__).parameters
        tfc_tdf_net_kwargs = dict((name, kwargs[name]) for name in valid_kwargs if name in kwargs)
        tfc_tdf_net_kwargs['n_fft'] = n_fft

        conditional_spec2spec = DCUN_TFC_GPoCM(**tfc_tdf_net_kwargs)

        train_loss_ = get_conditional_loss(train_loss, n_fft, hop_length, **kwargs)
        val_loss_ = get_conditional_loss(val_loss, n_fft, hop_length, **kwargs)

        super(DCUN_TFC_GPoCM_Framework, self).__init__(n_fft, hop_length, num_frame,
                                                       spec_type, spec_est_mode,
                                                       conditional_spec2spec,
                                                       optimizer, lr, dev_mode,
                                                       train_loss_, val_loss_
                                                       )

        valid_kwargs = inspect.signature(DCUN_TFC_GPoCM_Framework.__init__).parameters
        hp = [key for key in valid_kwargs.keys() if key not in ['self', 'kwargs']]
        hp = hp + [key for key in kwargs if not callable(kwargs[key])]
        self.save_hyperparameters(*hp)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_internal_layers', type=int, default=5)
        parser.add_argument('--kernel_size_t', type=int, default=3)
        parser.add_argument('--kernel_size_f', type=int, default=3)
        parser.add_argument('--tfc_activation', type=str, default='relu')

        return DenseCUNet_GPoCM_Framework.add_model_specific_args(parser)
