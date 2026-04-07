import math
import random

import gradio as gr
import torch

from modules import scripts
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks
from modules.ui_components import InputAccordion


def Fourier_filter(x, threshold, scale):
    x_freq = torch.fft.fftn(x.float(), dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W), device=x.device)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


def patch_FUM_v2(unet_patcher, b1, b2, s1, s2):
    model_channels = unet_patcher.model.diffusion_model.config.get("model_channels")

    scale_dict = {model_channels * 4: (b1, s1), model_channels * 2: (b2, s2)}
    on_cpu_devices = {}

    def output_block_patch(h, hsp, transformer_options):
        process = FUMForForge.doFUM

        if process:
            scale = scale_dict.get(h.shape[1], None)
            if scale is not None:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                denom = (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
                denom = torch.where(denom == 0, torch.ones_like(denom), denom)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / denom

                h[:, :h.shape[1] // 2] = h[:, :h.shape[1] // 2] * ((scale[0] - 1) * hidden_mean + 1)

                if hsp.device not in on_cpu_devices:
                    try:
                        hsp = Fourier_filter(hsp, threshold=1, scale=scale[1])
                    except Exception:
                        print("Device", hsp.device, "does not support the torch.fft functions used in the FUM node, switching to CPU.")
                        on_cpu_devices[hsp.device] = True
                        hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)
                else:
                    hsp = Fourier_filter(hsp.cpu(), threshold=1, scale=scale[1]).to(hsp.device)

        return h, hsp

    m = unet_patcher.clone()
    m.set_model_output_block_patch(output_block_patch)
    return m


def clamp_float(value, minimum, maximum, fallback=None):
    if fallback is None:
        fallback = minimum

    try:
        value = float(value)
    except (TypeError, ValueError):
        value = fallback

    if minimum > maximum:
        minimum, maximum = maximum, minimum

    return max(minimum, min(maximum, value))


def clamp_int(value, minimum, maximum, fallback=None):
    if fallback is None:
        fallback = minimum

    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        value = fallback

    if minimum > maximum:
        minimum, maximum = maximum, minimum

    return max(minimum, min(maximum, value))


def normalize_bounds(lower, upper, hard_min, hard_max, default_lower=None, default_upper=None):
    if default_lower is None:
        default_lower = hard_min
    if default_upper is None:
        default_upper = hard_max

    lower = clamp_float(lower, hard_min, hard_max, default_lower)
    upper = clamp_float(upper, hard_min, hard_max, default_upper)

    if lower > upper:
        lower, upper = upper, lower

    return lower, upper


def triangle_wave(t):
    t = float(t) % 1.0
    return 1.0 - abs((2.0 * t) - 1.0)


def cosine_ease(x):
    x = clamp_float(x, 0.0, 1.0, 0.0)
    return 0.5 - 0.5 * math.cos(x * math.pi)


class FUMForForge(scripts.Script):
    sorting_priority = 12

    doFUM = True
    FUM_start = 0.0
    FUM_end = 1.0

    last_b1 = 1.01
    last_b2 = 1.02
    last_s1 = 0.99
    last_s2 = 0.95

    motion_presets = [
        "Off",
        "Random Step",
        "Linear Up",
        "Linear Down",
        "YoYo",
        "Wave",
        "Triangle",
        "Saw Up",
        "Saw Down",
        "Pulse Return",
        "Ease In-Out",
        "Square Hold",
    ]
    motion_state = {}

    prompt_stabilizer_default = "plain black backdrop pattern."

    sequence_frame = 0
    sequence_initialized = False
    last_processing_id = None
    sequence_signature = None

    presets_builtin = [
        ('Forge default', 1.01, 1.02, 0.99, 0.95, 0.0, 1.0),
        ('SD 1.4', 1.3, 1.4, 0.9, 0.2, 0.0, 1.0),
        ('SD 1.5', 1.5, 1.6, 0.9, 0.2, 0.0, 1.0),
        ('SD 2.1', 1.4, 1.6, 0.9, 0.2, 0.0, 1.0),
        ('SDXL', 1.3, 1.4, 0.9, 0.2, 0.0, 1.0),
        ('PONY', 1.35, 1.44, 0.9, 0.2, 0.0, 1.0),
        ('Testing', 1.37, 1.42, 0.9, 0.2, 0.0, 1.0),
    ]
    try:
        import FUM_presets
        presets = presets_builtin + FUM_presets.presets_custom
    except Exception:
        presets = presets_builtin

    param_meta = {
        "b1": {"label": "B1", "minimum": 0.0, "maximum": 2.0, "default": 1.01},
        "b2": {"label": "B2", "minimum": 0.0, "maximum": 2.0, "default": 1.02},
        "s1": {"label": "S1", "minimum": 0.0, "maximum": 4.0, "default": 0.99},
        "s2": {"label": "S2", "minimum": 0.0, "maximum": 4.0, "default": 0.95},
    }

    def title(self):
        return "FUM (FreeU-Move)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def append_prompt_text(base_prompt, addition):
        addition = str(addition or "").strip()
        if not addition:
            return base_prompt

        prompt = str(base_prompt or "").rstrip()
        if not prompt:
            return addition

        if addition in prompt:
            return prompt

        separator = ""
        if prompt and not prompt.endswith((" ", ",", ";", ".", ":")):
            separator = " "

        return f"{prompt}{separator}{addition}"

    @classmethod
    def apply_prompt_stabilizer(cls, p, enabled, addition_text):
        if not enabled:
            return

        addition_text = str(addition_text or "").strip()
        if not addition_text:
            return

        if hasattr(p, "prompt"):
            p.prompt = cls.append_prompt_text(p.prompt, addition_text)

        if hasattr(p, "all_prompts") and isinstance(p.all_prompts, list):
            p.all_prompts = [cls.append_prompt_text(prompt, addition_text) for prompt in p.all_prompts]

        if hasattr(p, "hr_prompt") and isinstance(p.hr_prompt, str):
            p.hr_prompt = cls.append_prompt_text(p.hr_prompt, addition_text)

        if hasattr(p, "all_hr_prompts") and isinstance(p.all_hr_prompts, list):
            p.all_hr_prompts = [cls.append_prompt_text(prompt, addition_text) for prompt in p.all_hr_prompts]

    @classmethod
    def reset_motion_state(cls, name, origin, lower, upper, preset):
        origin = clamp_float(origin, lower, upper, origin)
        cls.motion_state[name] = {
            "current": origin,
            "direction": 1,
            "origin": origin,
            "lower": lower,
            "upper": upper,
            "preset": preset,
        }
        return cls.motion_state[name]

    @classmethod
    def get_sequence_frame(cls, p, start_frame, frame_step, auto_advance, reset_on_new_job):
        start_frame = clamp_int(start_frame, 0, 100000000, 0)
        frame_step = clamp_int(frame_step, 1, 100000000, 1)
        auto_advance = bool(auto_advance)
        reset_on_new_job = bool(reset_on_new_job)
        signature = (start_frame, frame_step, auto_advance, reset_on_new_job)
        processing_id = id(p)

        if cls.sequence_signature != signature:
            cls.sequence_signature = signature
            cls.sequence_frame = start_frame
            cls.sequence_initialized = True
            cls.last_processing_id = processing_id
            cls.motion_state.clear()
            return cls.sequence_frame

        if not cls.sequence_initialized:
            cls.sequence_frame = start_frame
            cls.sequence_initialized = True
            cls.last_processing_id = processing_id
            cls.motion_state.clear()
            return cls.sequence_frame

        if cls.last_processing_id != processing_id:
            if reset_on_new_job:
                cls.sequence_frame = start_frame
                cls.motion_state.clear()
            elif auto_advance:
                cls.sequence_frame += frame_step
            else:
                cls.sequence_frame = start_frame
                cls.motion_state.clear()
            cls.last_processing_id = processing_id

        return clamp_int(cls.sequence_frame, 0, 100000000, 0)

    @staticmethod
    def get_period_position(frame_index, motion_speed, cycle_frames, phase_offset):
        frame_index = clamp_int(frame_index, 0, 100000000, 0)
        motion_speed = clamp_float(motion_speed, 0.1, 10.0, 1.0)
        cycle_frames = clamp_int(cycle_frames, 1, 100000, 60)
        phase_offset = clamp_float(phase_offset, 0.0, 1.0, 0.0)
        return (((frame_index * motion_speed) / float(cycle_frames)) + phase_offset) % 1.0

    @classmethod
    def get_motion_value(cls, name, base_value, step_size, motion_speed, lower, upper, preset, frame_index, cycle_frames, phase_offset):
        base_value = clamp_float(base_value, lower, upper, base_value)

        if preset == "Off" or upper <= lower:
            cls.reset_motion_state(name, base_value, lower, upper, preset)
            return base_value

        state = cls.motion_state.get(name)
        if state is None:
            state = cls.reset_motion_state(name, base_value, lower, upper, preset)
        else:
            if (
                state.get("preset") != preset
                or abs(state.get("origin", base_value) - base_value) > 1e-12
                or abs(state.get("lower", lower) - lower) > 1e-12
                or abs(state.get("upper", upper) - upper) > 1e-12
            ):
                state = cls.reset_motion_state(name, base_value, lower, upper, preset)

        step_size = clamp_float(step_size, 0.0001, 0.1, 0.01)
        motion_speed = clamp_float(motion_speed, 0.1, 10.0, 1.0)
        cycle_frames = clamp_int(cycle_frames, 1, 100000, 60)
        phase_offset = clamp_float(phase_offset, 0.0, 1.0, 0.0)
        effective_step = max(0.0001, step_size * motion_speed)
        origin = clamp_float(base_value, lower, upper, base_value)
        span = upper - lower

        if preset == "Random Step":
            current = state["current"] + random.choice([-effective_step, effective_step])
            current = clamp_float(current, lower, upper, origin)
            state["current"] = current
            return current

        if preset == "Linear Up":
            current = origin + (effective_step * frame_index)
            current = clamp_float(current, lower, upper, origin)
            state["current"] = current
            return current

        if preset == "Linear Down":
            current = origin - (effective_step * frame_index)
            current = clamp_float(current, lower, upper, origin)
            state["current"] = current
            return current

        if preset == "YoYo":
            current = state["current"] + (state["direction"] * effective_step)
            if current >= upper:
                current = upper
                state["direction"] = -1
            elif current <= lower:
                current = lower
                state["direction"] = 1
            state["current"] = current
            return current

        t = cls.get_period_position(frame_index, motion_speed, cycle_frames, phase_offset)

        if preset == "Wave":
            normalized = 0.5 + 0.5 * math.sin((2.0 * math.pi * t) - (math.pi / 2.0))
            current = lower + (span * normalized)
            state["current"] = current
            return clamp_float(current, lower, upper, origin)

        if preset == "Triangle":
            normalized = triangle_wave(t)
            current = lower + (span * normalized)
            state["current"] = current
            return clamp_float(current, lower, upper, origin)

        if preset == "Saw Up":
            normalized = t
            current = lower + (span * normalized)
            state["current"] = current
            return clamp_float(current, lower, upper, origin)

        if preset == "Saw Down":
            normalized = 1.0 - t
            current = lower + (span * normalized)
            state["current"] = current
            return clamp_float(current, lower, upper, origin)

        if preset == "Pulse Return":
            target = upper if abs(upper - origin) >= abs(origin - lower) else lower
            normalized = cosine_ease(triangle_wave(t))
            current = origin + ((target - origin) * normalized)
            state["current"] = current
            return clamp_float(current, lower, upper, origin)

        if preset == "Ease In-Out":
            normalized = cosine_ease(triangle_wave(t))
            current = lower + (span * normalized)
            state["current"] = current
            return clamp_float(current, lower, upper, origin)

        if preset == "Square Hold":
            normalized = 1.0 if t < 0.5 else 0.0
            current = lower + (span * normalized)
            state["current"] = current
            return clamp_float(current, lower, upper, origin)

        return base_value

    def ui(self, *args, **kwargs):
        def sync_slider_to_number(minimum, maximum, digits=4):
            def _sync(value):
                return round(clamp_float(value, minimum, maximum, minimum), digits)
            return _sync

        def sync_number_to_slider(minimum, maximum, fallback, digits=4):
            def _sync(value):
                value = clamp_float(value, minimum, maximum, fallback)
                return value, round(value, digits)
            return _sync

        def sync_slider_to_int_number(minimum, maximum):
            def _sync(value):
                return clamp_int(value, minimum, maximum, minimum)
            return _sync

        def sync_int_number_to_slider(minimum, maximum, fallback):
            def _sync(value):
                value = clamp_int(value, minimum, maximum, fallback)
                return value, value
            return _sync

        def build_param_controls(param_name):
            meta = FUMForForge.param_meta[param_name]
            label = meta["label"]
            minimum = meta["minimum"]
            maximum = meta["maximum"]
            default = meta["default"]

            with gr.Group():
                with gr.Row():
                    value_slider = gr.Slider(label=label, minimum=minimum, maximum=maximum, step=0.01, value=default)
                    preset_dropdown = gr.Dropdown(label=f"{label} motion preset", choices=FUMForForge.motion_presets, value="Off")
                with gr.Row():
                    step_slider = gr.Slider(label=f"{label} move step", minimum=0.0001, maximum=0.1, step=0.0001, value=0.01)
                    step_number = gr.Number(label=f"{label} step exact", value=0.01, precision=4)
                    speed_slider = gr.Slider(label=f"{label} motion speed", minimum=0.1, maximum=10.0, step=0.1, value=1.0)
                    speed_number = gr.Number(label=f"{label} speed exact", value=1.0, precision=4)
                with gr.Row():
                    min_slider = gr.Slider(label=f"{label} lower bound", minimum=minimum, maximum=maximum, step=0.0001, value=minimum)
                    min_number = gr.Number(label=f"{label} lower exact", value=minimum, precision=4)
                    max_slider = gr.Slider(label=f"{label} upper bound", minimum=minimum, maximum=maximum, step=0.0001, value=maximum)
                    max_number = gr.Number(label=f"{label} upper exact", value=maximum, precision=4)
                with gr.Row():
                    cycle_slider = gr.Slider(label=f"{label} cycle frames", minimum=1, maximum=600, step=1, value=60)
                    cycle_number = gr.Number(label=f"{label} cycle exact", value=60, precision=0)
                    phase_slider = gr.Slider(label=f"{label} phase offset", minimum=0.0, maximum=1.0, step=0.001, value=0.0)
                    phase_number = gr.Number(label=f"{label} phase exact", value=0.0, precision=4)

            step_slider.input(fn=sync_slider_to_number(0.0001, 0.1), inputs=[step_slider], outputs=[step_number], show_progress=False)
            step_number.change(fn=sync_number_to_slider(0.0001, 0.1, 0.01), inputs=[step_number], outputs=[step_slider, step_number], show_progress=False)

            speed_slider.input(fn=sync_slider_to_number(0.1, 10.0), inputs=[speed_slider], outputs=[speed_number], show_progress=False)
            speed_number.change(fn=sync_number_to_slider(0.1, 10.0, 1.0), inputs=[speed_number], outputs=[speed_slider, speed_number], show_progress=False)

            min_slider.input(fn=sync_slider_to_number(minimum, maximum), inputs=[min_slider], outputs=[min_number], show_progress=False)
            min_number.change(fn=sync_number_to_slider(minimum, maximum, minimum), inputs=[min_number], outputs=[min_slider, min_number], show_progress=False)

            max_slider.input(fn=sync_slider_to_number(minimum, maximum), inputs=[max_slider], outputs=[max_number], show_progress=False)
            max_number.change(fn=sync_number_to_slider(minimum, maximum, maximum), inputs=[max_number], outputs=[max_slider, max_number], show_progress=False)

            cycle_slider.input(fn=sync_slider_to_int_number(1, 600), inputs=[cycle_slider], outputs=[cycle_number], show_progress=False)
            cycle_number.change(fn=sync_int_number_to_slider(1, 600, 60), inputs=[cycle_number], outputs=[cycle_slider, cycle_number], show_progress=False)

            phase_slider.input(fn=sync_slider_to_number(0.0, 1.0), inputs=[phase_slider], outputs=[phase_number], show_progress=False)
            phase_number.change(fn=sync_number_to_slider(0.0, 1.0, 0.0), inputs=[phase_number], outputs=[phase_slider, phase_number], show_progress=False)

            return {
                "value": value_slider,
                "preset": preset_dropdown,
                "step": step_slider,
                "step_number": step_number,
                "speed": speed_slider,
                "speed_number": speed_number,
                "min": min_slider,
                "min_number": min_number,
                "max": max_slider,
                "max_number": max_number,
                "cycle": cycle_slider,
                "cycle_number": cycle_number,
                "phase": phase_slider,
                "phase_number": phase_number,
            }

        with InputAccordion(False, label=self.title(), elem_id="extensions-FUM", elem_classes=["extensions-FUM"]) as FUM_enabled:
            controls = {}
            controls["b1"] = build_param_controls("b1")
            gr.HTML("<br><br>")
            controls["b2"] = build_param_controls("b2")
            gr.HTML("<br><br>")
            controls["s1"] = build_param_controls("s1")
            gr.HTML("<br><br>")
            controls["s2"] = build_param_controls("s2")
            gr.HTML("<br>")

            with gr.Row():
                FUM_start = gr.Slider(label='Start step', minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                FUM_end = gr.Slider(label='End step', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
            with gr.Row():
                FUM_preset = gr.Dropdown(label='', choices=[x[0] for x in FUMForForge.presets], value='(presets)', type='index', scale=0, allow_custom_value=True)
            with gr.Row():
                sequence_start = gr.Number(label="Sequence start frame", value=0, precision=0)
                sequence_step = gr.Number(label="Sequence frame step", value=1, precision=0)
                sequence_auto_advance = gr.Checkbox(label="Auto advance sequence frame", value=True)
                sequence_reset_on_new_job = gr.Checkbox(label="Reset to start frame on new job", value=False)
            with gr.Row():
                prompt_stabilizer_enabled = gr.Checkbox(label="Append prompt stabilizer text", value=False)
            with gr.Row():
                prompt_stabilizer_text = gr.Textbox(label="Prompt stabilizer text", value=FUMForForge.prompt_stabilizer_default, lines=1)

        def setParams(preset):
            if preset < len(FUMForForge.presets):
                return (
                    FUMForForge.presets[preset][1],
                    FUMForForge.presets[preset][2],
                    FUMForForge.presets[preset][3],
                    FUMForForge.presets[preset][4],
                    FUMForForge.presets[preset][5],
                    FUMForForge.presets[preset][6],
                    '(presets)',
                )
            return 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, '(presets)'

        FUM_preset.input(
            fn=setParams,
            inputs=[FUM_preset],
            outputs=[
                controls["b1"]["value"],
                controls["b2"]["value"],
                controls["s1"]["value"],
                controls["s2"]["value"],
                FUM_start,
                FUM_end,
                FUM_preset,
            ],
            show_progress=False,
        )

        self.infotext_fields = [
            (FUM_enabled, lambda d: d.get("FUM_enabled", False)),
            (controls["b1"]["value"], "FUM_b1"),
            (controls["b2"]["value"], "FUM_b2"),
            (controls["s1"]["value"], "FUM_s1"),
            (controls["s2"]["value"], "FUM_s2"),
            (controls["b1"]["preset"], "FUM_b1_preset"),
            (controls["b2"]["preset"], "FUM_b2_preset"),
            (controls["s1"]["preset"], "FUM_s1_preset"),
            (controls["s2"]["preset"], "FUM_s2_preset"),
            (controls["b1"]["step"], "FUM_b1_step"),
            (controls["b2"]["step"], "FUM_b2_step"),
            (controls["s1"]["step"], "FUM_s1_step"),
            (controls["s2"]["step"], "FUM_s2_step"),
            (controls["b1"]["speed"], "FUM_b1_speed"),
            (controls["b2"]["speed"], "FUM_b2_speed"),
            (controls["s1"]["speed"], "FUM_s1_speed"),
            (controls["s2"]["speed"], "FUM_s2_speed"),
            (controls["b1"]["min"], "FUM_b1_min"),
            (controls["b2"]["min"], "FUM_b2_min"),
            (controls["s1"]["min"], "FUM_s1_min"),
            (controls["s2"]["min"], "FUM_s2_min"),
            (controls["b1"]["max"], "FUM_b1_max"),
            (controls["b2"]["max"], "FUM_b2_max"),
            (controls["s1"]["max"], "FUM_s1_max"),
            (controls["s2"]["max"], "FUM_s2_max"),
            (controls["b1"]["cycle"], "FUM_b1_cycle"),
            (controls["b2"]["cycle"], "FUM_b2_cycle"),
            (controls["s1"]["cycle"], "FUM_s1_cycle"),
            (controls["s2"]["cycle"], "FUM_s2_cycle"),
            (controls["b1"]["phase"], "FUM_b1_phase"),
            (controls["b2"]["phase"], "FUM_b2_phase"),
            (controls["s1"]["phase"], "FUM_s1_phase"),
            (controls["s2"]["phase"], "FUM_s2_phase"),
            (FUM_start, "FUM_start"),
            (FUM_end, "FUM_end"),
            (sequence_start, "FUM_sequence_start"),
            (sequence_step, "FUM_sequence_step"),
            (sequence_auto_advance, lambda d: d.get("FUM_sequence_auto_advance", True)),
            (sequence_reset_on_new_job, lambda d: d.get("FUM_sequence_reset_on_new_job", False)),
            (prompt_stabilizer_enabled, lambda d: d.get("FUM_prompt_stabilizer_enabled", False)),
            (prompt_stabilizer_text, "FUM_prompt_stabilizer_text"),
        ]

        return (
            FUM_enabled,
            controls["b1"]["value"], controls["b1"]["preset"], controls["b1"]["step"], controls["b1"]["step_number"], controls["b1"]["speed"], controls["b1"]["speed_number"], controls["b1"]["min"], controls["b1"]["min_number"], controls["b1"]["max"], controls["b1"]["max_number"], controls["b1"]["cycle"], controls["b1"]["cycle_number"], controls["b1"]["phase"], controls["b1"]["phase_number"],
            controls["b2"]["value"], controls["b2"]["preset"], controls["b2"]["step"], controls["b2"]["step_number"], controls["b2"]["speed"], controls["b2"]["speed_number"], controls["b2"]["min"], controls["b2"]["min_number"], controls["b2"]["max"], controls["b2"]["max_number"], controls["b2"]["cycle"], controls["b2"]["cycle_number"], controls["b2"]["phase"], controls["b2"]["phase_number"],
            controls["s1"]["value"], controls["s1"]["preset"], controls["s1"]["step"], controls["s1"]["step_number"], controls["s1"]["speed"], controls["s1"]["speed_number"], controls["s1"]["min"], controls["s1"]["min_number"], controls["s1"]["max"], controls["s1"]["max_number"], controls["s1"]["cycle"], controls["s1"]["cycle_number"], controls["s1"]["phase"], controls["s1"]["phase_number"],
            controls["s2"]["value"], controls["s2"]["preset"], controls["s2"]["step"], controls["s2"]["step_number"], controls["s2"]["speed"], controls["s2"]["speed_number"], controls["s2"]["min"], controls["s2"]["min_number"], controls["s2"]["max"], controls["s2"]["max_number"], controls["s2"]["cycle"], controls["s2"]["cycle_number"], controls["s2"]["phase"], controls["s2"]["phase_number"],
            FUM_start, FUM_end,
            sequence_start, sequence_step, sequence_auto_advance, sequence_reset_on_new_job,
            prompt_stabilizer_enabled, prompt_stabilizer_text,
        )

    def denoiser_callback(self, params):
        if params.total_sampling_steps <= 1:
            thisStep = 0.0
        else:
            thisStep = params.sampling_step / (params.total_sampling_steps - 1)

        FUMForForge.doFUM = FUMForForge.FUM_start <= thisStep <= FUMForForge.FUM_end

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (
            FUM_enabled,
            FUM_b1, FUM_b1_preset, FUM_b1_step, FUM_b1_step_number, FUM_b1_speed, FUM_b1_speed_number, FUM_b1_min, FUM_b1_min_number, FUM_b1_max, FUM_b1_max_number, FUM_b1_cycle, FUM_b1_cycle_number, FUM_b1_phase, FUM_b1_phase_number,
            FUM_b2, FUM_b2_preset, FUM_b2_step, FUM_b2_step_number, FUM_b2_speed, FUM_b2_speed_number, FUM_b2_min, FUM_b2_min_number, FUM_b2_max, FUM_b2_max_number, FUM_b2_cycle, FUM_b2_cycle_number, FUM_b2_phase, FUM_b2_phase_number,
            FUM_s1, FUM_s1_preset, FUM_s1_step, FUM_s1_step_number, FUM_s1_speed, FUM_s1_speed_number, FUM_s1_min, FUM_s1_min_number, FUM_s1_max, FUM_s1_max_number, FUM_s1_cycle, FUM_s1_cycle_number, FUM_s1_phase, FUM_s1_phase_number,
            FUM_s2, FUM_s2_preset, FUM_s2_step, FUM_s2_step_number, FUM_s2_speed, FUM_s2_speed_number, FUM_s2_min, FUM_s2_min_number, FUM_s2_max, FUM_s2_max_number, FUM_s2_cycle, FUM_s2_cycle_number, FUM_s2_phase, FUM_s2_phase_number,
            FUM_start, FUM_end,
            sequence_start, sequence_step, sequence_auto_advance, sequence_reset_on_new_job,
            prompt_stabilizer_enabled, prompt_stabilizer_text,
        ) = script_args

        if not FUM_enabled:
            return

        FUMForForge.apply_prompt_stabilizer(p, prompt_stabilizer_enabled, prompt_stabilizer_text)

        frame_index = FUMForForge.get_sequence_frame(
            p,
            sequence_start,
            sequence_step,
            sequence_auto_advance,
            sequence_reset_on_new_job,
        )

        configs = {
            "b1": {
                "value": FUM_b1,
                "preset": FUM_b1_preset,
                "step": FUM_b1_step_number if FUM_b1_step_number is not None else FUM_b1_step,
                "speed": FUM_b1_speed_number if FUM_b1_speed_number is not None else FUM_b1_speed,
                "min": FUM_b1_min_number if FUM_b1_min_number is not None else FUM_b1_min,
                "max": FUM_b1_max_number if FUM_b1_max_number is not None else FUM_b1_max,
                "cycle": FUM_b1_cycle_number if FUM_b1_cycle_number is not None else FUM_b1_cycle,
                "phase": FUM_b1_phase_number if FUM_b1_phase_number is not None else FUM_b1_phase,
            },
            "b2": {
                "value": FUM_b2,
                "preset": FUM_b2_preset,
                "step": FUM_b2_step_number if FUM_b2_step_number is not None else FUM_b2_step,
                "speed": FUM_b2_speed_number if FUM_b2_speed_number is not None else FUM_b2_speed,
                "min": FUM_b2_min_number if FUM_b2_min_number is not None else FUM_b2_min,
                "max": FUM_b2_max_number if FUM_b2_max_number is not None else FUM_b2_max,
                "cycle": FUM_b2_cycle_number if FUM_b2_cycle_number is not None else FUM_b2_cycle,
                "phase": FUM_b2_phase_number if FUM_b2_phase_number is not None else FUM_b2_phase,
            },
            "s1": {
                "value": FUM_s1,
                "preset": FUM_s1_preset,
                "step": FUM_s1_step_number if FUM_s1_step_number is not None else FUM_s1_step,
                "speed": FUM_s1_speed_number if FUM_s1_speed_number is not None else FUM_s1_speed,
                "min": FUM_s1_min_number if FUM_s1_min_number is not None else FUM_s1_min,
                "max": FUM_s1_max_number if FUM_s1_max_number is not None else FUM_s1_max,
                "cycle": FUM_s1_cycle_number if FUM_s1_cycle_number is not None else FUM_s1_cycle,
                "phase": FUM_s1_phase_number if FUM_s1_phase_number is not None else FUM_s1_phase,
            },
            "s2": {
                "value": FUM_s2,
                "preset": FUM_s2_preset,
                "step": FUM_s2_step_number if FUM_s2_step_number is not None else FUM_s2_step,
                "speed": FUM_s2_speed_number if FUM_s2_speed_number is not None else FUM_s2_speed,
                "min": FUM_s2_min_number if FUM_s2_min_number is not None else FUM_s2_min,
                "max": FUM_s2_max_number if FUM_s2_max_number is not None else FUM_s2_max,
                "cycle": FUM_s2_cycle_number if FUM_s2_cycle_number is not None else FUM_s2_cycle,
                "phase": FUM_s2_phase_number if FUM_s2_phase_number is not None else FUM_s2_phase,
            },
        }

        for name, cfg in configs.items():
            meta = FUMForForge.param_meta[name]
            hard_min = meta["minimum"]
            hard_max = meta["maximum"]
            lower, upper = normalize_bounds(cfg["min"], cfg["max"], hard_min, hard_max, hard_min, hard_max)
            cfg["min"] = lower
            cfg["max"] = upper
            cfg["value"] = clamp_float(cfg["value"], hard_min, hard_max, meta["default"])
            cfg["step"] = clamp_float(cfg["step"], 0.0001, 0.1, 0.01)
            cfg["speed"] = clamp_float(cfg["speed"], 0.1, 10.0, 1.0)
            cfg["cycle"] = clamp_int(cfg["cycle"], 1, 100000, 60)
            cfg["phase"] = clamp_float(cfg["phase"], 0.0, 1.0, 0.0)

            cfg["value"] = FUMForForge.get_motion_value(
                name=name,
                base_value=cfg["value"],
                step_size=cfg["step"],
                motion_speed=cfg["speed"],
                lower=lower,
                upper=upper,
                preset=cfg["preset"],
                frame_index=frame_index,
                cycle_frames=cfg["cycle"],
                phase_offset=cfg["phase"],
            )

        FUMForForge.last_b1 = configs["b1"]["value"]
        FUMForForge.last_b2 = configs["b2"]["value"]
        FUMForForge.last_s1 = configs["s1"]["value"]
        FUMForForge.last_s2 = configs["s2"]["value"]

        unet = p.sd_model.forge_objects.unet
        model_channels = unet.model.diffusion_model.config.get("model_channels")
        if model_channels is None:
            gr.Info("FUM is not supported for this model!")
            return

        FUMForForge.FUM_start = clamp_float(FUM_start, 0.0, 1.0, 0.0)
        FUMForForge.FUM_end = clamp_float(FUM_end, 0.0, 1.0, 1.0)
        if FUMForForge.FUM_start > FUMForForge.FUM_end:
            FUMForForge.FUM_start, FUMForForge.FUM_end = FUMForForge.FUM_end, FUMForForge.FUM_start

        on_cfg_denoiser(self.denoiser_callback)

        unet = patch_FUM_v2(
            unet,
            FUMForForge.last_b1,
            FUMForForge.last_b2,
            FUMForForge.last_s1,
            FUMForForge.last_s2,
        )

        p.sd_model.forge_objects.unet = unet

        p.extra_generation_params.update(dict(
            FUM_enabled=FUM_enabled,
            FUM_b1=FUMForForge.last_b1,
            FUM_b2=FUMForForge.last_b2,
            FUM_s1=FUMForForge.last_s1,
            FUM_s2=FUMForForge.last_s2,
            FUM_b1_preset=FUM_b1_preset,
            FUM_b2_preset=FUM_b2_preset,
            FUM_s1_preset=FUM_s1_preset,
            FUM_s2_preset=FUM_s2_preset,
            FUM_b1_step=configs["b1"]["step"],
            FUM_b2_step=configs["b2"]["step"],
            FUM_s1_step=configs["s1"]["step"],
            FUM_s2_step=configs["s2"]["step"],
            FUM_b1_speed=configs["b1"]["speed"],
            FUM_b2_speed=configs["b2"]["speed"],
            FUM_s1_speed=configs["s1"]["speed"],
            FUM_s2_speed=configs["s2"]["speed"],
            FUM_b1_min=configs["b1"]["min"],
            FUM_b2_min=configs["b2"]["min"],
            FUM_s1_min=configs["s1"]["min"],
            FUM_s2_min=configs["s2"]["min"],
            FUM_b1_max=configs["b1"]["max"],
            FUM_b2_max=configs["b2"]["max"],
            FUM_s1_max=configs["s1"]["max"],
            FUM_s2_max=configs["s2"]["max"],
            FUM_b1_cycle=configs["b1"]["cycle"],
            FUM_b2_cycle=configs["b2"]["cycle"],
            FUM_s1_cycle=configs["s1"]["cycle"],
            FUM_s2_cycle=configs["s2"]["cycle"],
            FUM_b1_phase=configs["b1"]["phase"],
            FUM_b2_phase=configs["b2"]["phase"],
            FUM_s1_phase=configs["s1"]["phase"],
            FUM_s2_phase=configs["s2"]["phase"],
            FUM_start=FUMForForge.FUM_start,
            FUM_end=FUMForForge.FUM_end,
            FUM_sequence_frame=frame_index,
            FUM_sequence_start=clamp_int(sequence_start, 0, 100000000, 0),
            FUM_sequence_step=clamp_int(sequence_step, 1, 100000000, 1),
            FUM_sequence_auto_advance=bool(sequence_auto_advance),
            FUM_sequence_reset_on_new_job=bool(sequence_reset_on_new_job),
            FUM_prompt_stabilizer_enabled=bool(prompt_stabilizer_enabled),
            FUM_prompt_stabilizer_text=str(prompt_stabilizer_text or "").strip(),
        ))

    def postprocess(self, params, processed, *args):
        remove_current_script_callbacks()
        return
