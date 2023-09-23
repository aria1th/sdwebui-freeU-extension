from modules import scripts, script_callbacks, shared
from modules.sd_hijack_utils import CondFunc
from scripts.freeunet_hijack import freeU_forward, cond_activate_freeu

# attach the option to the shared options
def on_ui_settings():
    shared.opts.add_option("freeu_enabled", shared.OptionInfo(True, "Free UNet Enabled", section=("freeu", "FreeUNet")))

script_callbacks.on_ui_settings(on_ui_settings)

CondFunc(
    'ldm.modules.diffusionmodules.openaimodel.UNetModel.forward',
    sub_func=lambda original_func, self, x, timesteps=None, context=None, y=None, **kwargs: freeU_forward(self, x, timesteps, context, y, **kwargs),
    cond_func=lambda original_func, self, x, timesteps=None, context=None, y=None, **kwargs: cond_activate_freeu(self, x, timesteps, context, y, **kwargs)
)