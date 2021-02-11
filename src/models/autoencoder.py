from .encoders import *
from .decoders import *

from .ablation.sparse_autoencoder import sparse_autoencoder
from .ablation.gaussian_blur import gaussian_blur


class autoencoder_class(nn.Module):
    def __init__(self, args):
        super(autoencoder_class, self).__init__()
        from .encoders import encoder_dict
        from .decoders import decoder_dict

        arch_name = args.autoencoder_arch.replace("_autoencoder", "")

        for decoder_name in ["small", "deep", "resize", "identity"]:
            if decoder_name in arch_name:
                decoder_class = decoder_name+"_decoder"
                encoder_class = arch_name.replace("_"+decoder_name)
                decoder_name = ""
                break

        if decoder_name != "":
            decoder_class = "default_decoder"
            encoder_class = arch_name

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
            return super(autoencoder_class, self).__getattr__(key)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def dictionary_update_off(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def dictionary_update_on(self):
        self.dictionary_update_off()
        self.encoder.conv.weight.requires_grad = True
