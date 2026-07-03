# Alpamayo Integration

This project can use NVIDIA Alpamayo as a non-controlling VLA teacher.
The stable driving stack still stays local:

- natural-language/action policy: `chat_gui_node`
- lane and zone control: `navigator_node`
- lane perception: existing YOLO/lane pipeline

Alpamayo should first be used to observe the same state and produce reasoning,
not to directly publish driving commands.

## Official Resources

- Alpamayo 1 / Alpamayo-R1 code: https://github.com/NVlabs/alpamayo
- Alpamayo 1 weights: https://huggingface.co/nvidia/Alpamayo-R1-10B
- Alpamayo 1.5 code: https://github.com/NVlabs/alpamayo1.5
- Alpamayo 1.5 weights: https://huggingface.co/nvidia/Alpamayo-1.5-10B

Alpamayo 1.5 is the better first target for this project because it supports
navigation guidance and VQA-style reasoning. The official README currently
expects a CUDA GPU with about 24 GB or more VRAM for single-sample inference.

## ROS GUI Hook

`chat_gui_node` has an optional Alpamayo teacher mode. In this mode the right
side VLA panel sends a compact ROS snapshot to an external HTTP endpoint and
shows the returned reasoning text.

For immediate connection testing, start the local Alpamayo-compatible adapter:

```bash
ros2 run nav_vla alpamayo_teacher_server
```

This adapter does not load the 10B Alpamayo model. It only validates the `/judge`
contract and returns teacher-style reasoning from the ROS snapshot. Replace it
with a real Alpamayo 1.5 inference server later.

```bash
ros2 run nav_vla chat_gui_node --ros-args \
  -p parser_backend:=action_policy \
  -p vla_judgment_backend:=alpamayo \
  -p alpamayo_endpoint:=http://127.0.0.1:8765/judge
```

The endpoint receives:

```json
{
  "model": "nvidia/Alpamayo-1.5-10B",
  "prompt": "Chain-of-Causation teacher prompt...",
  "snapshot": {
    "command": "go M3 through lane1",
    "parsed_steps": [],
    "current_lane": "lane2",
    "nav_status": "...",
    "detections": [],
    "lane_info": {},
    "path": {},
    "pose": {}
  }
}
```

The endpoint can return plain text or JSON with one of these fields:

```json
{
  "reasoning": "The parsed command targets M3 in lane1..."
}
```

Accepted JSON keys are `reasoning`, `judgment`, `coc`,
`chain_of_causation`, `text`, `output`, or `message`.

## Real Alpamayo 1.5 Server

Run the real model outside the ROS environment. The official Alpamayo 1.5
release uses Python 3.12, CUDA, Hugging Face gated access, and about 24 GB or
more VRAM for single-sample inference.

First check that the GPU driver is visible:

```bash
nvidia-smi
```

Set up Alpamayo 1.5 separately:

```bash
mkdir -p ~/ROS2_project/alpamayo_ws
cd ~/ROS2_project/alpamayo_ws
git clone https://github.com/NVlabs/alpamayo1.5.git
cd alpamayo1.5

# Install uv if needed:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv a1_5_venv
source a1_5_venv/bin/activate

# Use the official install first. If flash-attn fails, use the SDPA fallback.
uv sync --active
# fallback:
# uv sync --active --no-install-package flash-attn

hf auth login
```

Request access first:

- https://huggingface.co/nvidia/Alpamayo-1.5-10B
- https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles

Then start the real `/judge` server from the Alpamayo environment:

```bash
python ~/ROS2_project/nav-vla/src/nav_vla/nav_vla/alpamayo_real_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --model-id nvidia/Alpamayo-1.5-10B
```

If `flash-attn` was skipped, start it with SDPA:

```bash
python ~/ROS2_project/nav-vla/src/nav_vla/nav_vla/alpamayo_real_server.py \
  --host 127.0.0.1 \
  --port 8765 \
  --model-id nvidia/Alpamayo-1.5-10B \
  --attn-implementation sdpa
```

This server uses Alpamayo 1.5's VQA path first. It receives the latest ROS
camera frame plus the local navigation snapshot and returns teacher reasoning.
It does not publish ROS commands.

## Recommended Workflow

1. Keep normal driving on the stable local stack.
2. Run Alpamayo 1.5 in its own `uv`/Hugging Face environment, separate from ROS.
3. Wrap Alpamayo inference with a small HTTP `/judge` adapter.
4. Compare Alpamayo reasoning against the local action policy and navigator
   state in the GUI.
5. Only after the teacher is reliable, use its outputs as labels or auxiliary
   supervision for a smaller student model.
