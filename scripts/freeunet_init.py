from modules import scripts, script_callbacks
from modules.processing import StableDiffusionProcessing
from modules.sd_hijack_utils import CondFunc
from scripts.freeunet_hijack import freeU_forward, cond_activate_freeu, attach_freeu_parameter, detach_freeu_parameter
from scripts.freeunet_xyz import make_axis_options
import gradio as gr

CondFunc(
    'ldm.modules.diffusionmodules.openaimodel.UNetModel.forward',
    sub_func=lambda original_func, self, x, timesteps=None, context=None, y=None, **kwargs: freeU_forward(self, x, timesteps, context, y, **kwargs),
    cond_func=lambda original_func, self, x, timesteps=None, context=None, y=None, **kwargs: cond_activate_freeu(self, x, timesteps, context, y, **kwargs)
)

class Script(scripts.Script):

    def title(self):
        return "FreeUNet"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        enabled = gr.Checkbox(value=False, label="Enable FreeUNet")
        self.infotext_fields = (
            (enabled, lambda d: gr.Checkbox.update(value="FreeUNet Enabled" in d)),
        )
        return [enabled, #return related parameters
        ]
    
    def process_batch(self, p:StableDiffusionProcessing, enabled, # related parameters
                      *args, **kwargs): # other parameters
        enabled = getattr(p, "freeu_enabled", enabled)
        p.freeu_enabled = enabled
        openai_model = p.sd_model.model.diffusion_model
        if enabled:
            attach_freeu_parameter(openai_model)
        else:
            detach_freeu_parameter(openai_model)
        if not enabled:
            return
        p.extra_generation_params["Free UNet Enabled"] = p.freeu_enabled
        
script_callbacks.on_before_ui(make_axis_options)