import torch
from torch.fft import (
    fftn, ifftn, fftshift, ifftshift
)

from modules.sd_hijack_unet import th
# requires dependencies installed
from modules.devices import device
from modules.shared import opts
from scripts.settings import BaseFreeUParameter

from ldm.modules.diffusionmodules.openaimodel import timestep_embedding, UNetModel
# https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/5ef669de080814067961f28357256e8fe27544f4/modules/sd_hijack_unet.py#L72
# the function may have to be changed in runtime

def freeU_forward(self:UNetModel, x:torch.Tensor, timesteps=None, context=None, y=None,**kwargs):
    """
    OpenAIModel.forward with hijack for FreeU.
    Refer to https://github.com/ChenyangSi/FreeU for more details.
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """
    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    #print("FreeU is enabled.")
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape == (x.shape[0],)
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    module:torch.nn.Module = None
    for module in self.input_blocks:
        h = module(h, emb, context)
        hs.append(h)
    # middle block
    h = self.middle_block(h, emb, context)
    # output blocks
    for module in self.output_blocks:
        hs_ = hs.pop()
        # ===FreeU Code, only works for 1280 or 640 stages===
        freeu_parameter = self.freeu_parameter
        if h.shape[1] == 1280:
            h[:,:640] = h[:,:640] * freeu_parameter.b1
            hs_ = fourier_filter(hs_, threshold=1, scale=freeu_parameter.s1)
        elif h.shape[1] == 640:
            h[:,:320] = h[:,:320] * freeu_parameter.b2
            hs_ = fourier_filter(hs_, threshold=1, scale=freeu_parameter.s2)
        # ===End of FreeU Code===
        h = th.cat([h, hs_], dim=1)
        h = module(h, emb, context)
    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)
    
def cond_activate_freeu(self:UNetModel, *args, **kwargs):
    """
    Condition check for FreeU.
    """
    if opts.freeu_enabled:
        # attach freeu_parameter to UNetModel
        attach_freeu_parameter(self)
    return opts.freeu_enabled

def attach_freeu_parameter(self:UNetModel):
    # attach BaseFreeUParameter to UNetModel
    setattr(self, 'freeu_parameter', BaseFreeUParameter())

def fourier_filter(x:torch.Tensor, threshold:float, scale:float) -> torch.Tensor:
    """
    Fourier filter for FreeU.
    Apply a filter to the input tensor in the frequency domain.
    The center of the frequency domain is scaled by the scale parameter with a threshold.
    """
    # if dtype is not float32, cast to float32 because cuFFT only supports float32
    orig_dtype = x.dtype
    if x.dtype != torch.float32:
        x = x.float()
    # FFT
    x_fft = fftn(
        x, dim=(-2, -1)
    ) # fftn: N-Dimensional FFT
    x_freq = fftshift(
        x_fft, dim=(-2, -1)
    ) # Centering the zero-frequency component
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), dtype=torch.bool, device=device)
    
    center_rows, center_cols = H // 2, W // 2 # simple 1/2 centering
    mask[..., center_rows - threshold : center_rows,
         center_cols - threshold : center_cols] = scale # apply scale for center
    x_freq = x_freq * mask # apply mask
    
    # IFFT
    x_freq = ifftshift(
        x_freq, dim=(-2, -1)
    ) # Inverse of fftshift
    x_ifft = ifftn(
        x_freq, dim=(-2, -1)
    ) # Inverse of fftn
    real_part = x_ifft.real # return real part
    if orig_dtype != torch.float32:
        real_part = real_part.to(orig_dtype)
    return real_part
