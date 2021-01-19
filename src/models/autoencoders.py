from .encoders import *
from .decoders import *

from .ablation.sparse_autoencoder import sparse_autoencoder
from .ablation.gaussian_blur import gaussian_blur


class autoencoder_base_class(nn.Module):
    def __init__(self, args, encoder_class, decoder_class="default_decoder"):
        super(autoencoder_base_class, self).__init__()
        from .encoders import encoder_dict
        from .decoders import decoder_dict

        self.encoder = encoder_dict[encoder_class](args)
        self.decoder = decoder_dict[decoder_class](args)

        self.jump.requires_grad = False
        self.dictionary.requires_grad = False
        self.l1_norms.requires_grad = False

    def __getattr__(self, key):

        if (
            key == "jump"
            or key == "dictionary"
            or key == "T"
            or key == "p"
            or key == "l1_norms"
            or key == "set_BPDA_type"
            or key == "fix_seed"
        ):
            return getattr(self.encoder, key)
        else:
            return super(autoencoder_base_class, self).__getattr__(key)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encoder_no_update(self):
        for p in self.encoder.parameters():
            p.requires_grad = False


class quant_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(quant_autoencoder, self).__init__(args, "quant_encoder")


class top_T_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_autoencoder, self).__init__(args, "top_T_encoder")


class top_T_noisy_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_noisy_autoencoder, self).__init__(
            args, "top_T_noisy_encoder")


class top_T_quant_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_quant_autoencoder, self).__init__(
            args, "top_T_quant_encoder")


class top_T_dropout_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_dropout_autoencoder, self).__init__(
            args, "top_T_dropout_encoder"
        )


class top_T_dropout_quant_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_dropout_quant_autoencoder, self).__init__(
            args, "top_T_dropout_quant_encoder"
        )


class top_T_dropout_quant_small_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_dropout_quant_small_autoencoder, self).__init__(
            args, "top_T_dropout_quant_encoder", "small_decoder"
        )


class top_T_dropout_quant_deep_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_dropout_quant_deep_autoencoder, self).__init__(
            args, "top_T_dropout_quant_encoder", "deep_decoder"
        )


class top_T_dropout_quant_resize_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_dropout_quant_resize_autoencoder, self).__init__(
            args, "top_T_dropout_quant_encoder", "resize_decoder"
        )


class top_T_quant_resize_autoencoder(autoencoder_base_class):
    def __init__(self, args):
        super(top_T_quant_resize_autoencoder, self).__init__(
            args, "top_T_quant_encoder", "resize_decoder"
        )


autoencoder_dict = {
    "quant_autoencoder": quant_autoencoder,
    "top_T_autoencoder": top_T_autoencoder,
    "top_T_quant_autoencoder": top_T_quant_autoencoder,
    "top_T_dropout_autoencoder": top_T_dropout_autoencoder,
    "top_T_dropout_quant_autoencoder": top_T_dropout_quant_autoencoder,
    "top_T_dropout_quant_small_autoencoder": top_T_dropout_quant_small_autoencoder,
    "top_T_dropout_quant_deep_autoencoder": top_T_dropout_quant_deep_autoencoder,
    "top_T_dropout_quant_resize_autoencoder": top_T_dropout_quant_resize_autoencoder,
    "top_T_quant_resize_autoencoder": top_T_quant_resize_autoencoder,
    "sparse_autoencoder": sparse_autoencoder,
    "gaussian_blur": gaussian_blur,
}
