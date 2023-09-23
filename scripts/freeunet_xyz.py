from modules import scripts

def make_axis_options():
    xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
    
    def apply_freeu(p, x, xs):
        # x may come as a string
        x = x.lower()
        x = x in ["true", "1", "yes", "y"]
        if x:
            setattr(p, "freeu_enabled", True)
        else:
            setattr(p, "freeu_enabled", False)
    extra_axis_options = [
        xyz_grid.AxisOption("[FreeU] Enabled", str, apply_freeu) # string because gradio doesn't support bool
    ]
    if not any("[FreeU]" in x.label for x in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(extra_axis_options)
        