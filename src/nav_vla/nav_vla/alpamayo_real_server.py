"""HTTP /judge server backed by NVIDIA Alpamayo 1.5.

Run this from an environment where NVlabs/alpamayo1.5 is installed and the
gated Hugging Face model access has been approved. This server intentionally
does not publish ROS commands. It only returns VQA/CoC-style teacher reasoning
for chat_gui_node's right-side panel.
"""

import argparse
import base64
import io
import json
import traceback
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer


MODEL_ID = "nvidia/Alpamayo-1.5-10B"
COSMOS_BACKBONE_ID = "nvidia/Cosmos-Reason2-8B"


class AlpamayoRuntime:
    def __init__(
        self,
        model_id=MODEL_ID,
        device="cuda",
        dtype="bfloat16",
        attn_implementation=None,
        max_generation_length=256,
        temperature=0.6,
        top_p=0.98,
        num_frames_per_camera=4,
    ):
        self.model_id = model_id
        self.device = device
        self.max_generation_length = max_generation_length
        self.temperature = temperature
        self.top_p = top_p
        self.num_frames_per_camera = num_frames_per_camera

        try:
            import numpy as np
            import torch
            from PIL import Image as PILImage
            from alpamayo1_5 import helper
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        except ImportError as exc:
            raise RuntimeError(
                "Alpamayo dependencies are missing. Install NVlabs/alpamayo1.5 "
                "in a separate Python 3.12 CUDA environment first."
            ) from exc

        self.torch = torch
        self.np = np
        self.PILImage = PILImage
        self.helper = helper
        self.device_type = "cuda" if str(device).startswith("cuda") else str(device)

        torch_dtype = getattr(torch, dtype)
        kwargs = {"dtype": torch_dtype}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        print(f"loading Alpamayo model: {model_id}")
        try:
            self.model = Alpamayo1_5.from_pretrained(model_id, **kwargs).to(device)
        except Exception as exc:
            message = str(exc)
            if "Cosmos-Reason2-8B" in message or "gated repo" in message.lower():
                raise RuntimeError(
                    "Alpamayo 1.5 loaded its config, but the Cosmos backbone is "
                    "still gated for this Hugging Face account. Request access to "
                    f"https://huggingface.co/{COSMOS_BACKBONE_ID}, then rerun "
                    "`hf auth login` if needed."
                ) from exc
            raise
        self.model.eval()
        self.processor = helper.get_processor(self.model.tokenizer)
        print("Alpamayo model ready")

    def judge(self, payload):
        snapshot = payload.get("snapshot") or {}
        images = self._decode_images(snapshot.get("images") or [])
        if not images:
            return {
                "reasoning": (
                    "Alpamayo 1.5 is loaded, but no camera image was included in "
                    "the request. Start the simulator with the camera bridge and "
                    "check /camera/image_raw."
                )
            }

        question = self._build_question(snapshot)
        frames = self._images_to_frames(images)
        camera_indices = self.torch.tensor([1], dtype=self.torch.long)
        messages = self.helper.create_vqa_message(
            frames,
            question=question,
            camera_indices=camera_indices,
            num_frames_per_camera=self.num_frames_per_camera,
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {"tokenized_data": inputs}
        model_inputs = self.helper.to_device(model_inputs, self.device)
        with self.torch.no_grad(), self.torch.autocast(
            self.device_type,
            dtype=self.torch.bfloat16,
        ):
            extra = self.model.generate_text(
                data=model_inputs,
                top_p=self.top_p,
                temperature=self.temperature,
                num_samples=1,
                max_generation_length=self.max_generation_length,
            )
        answer = self._answer_to_text(extra.get("answer"))
        return {
            "reasoning": answer,
            "source": "nvidia_alpamayo_1_5",
            "model": self.model_id,
        }

    def _answer_to_text(self, answer):
        if isinstance(answer, str):
            return answer.strip()
        if isinstance(answer, bytes):
            return answer.decode("utf-8", errors="replace").strip()
        if hasattr(answer, "tolist"):
            answer = answer.tolist()
        if isinstance(answer, (list, tuple)):
            if len(answer) == 1:
                return self._answer_to_text(answer[0])
            return " ".join(self._answer_to_text(item) for item in answer).strip()
        if answer is None:
            return "Alpamayo returned no textual answer."
        return str(answer).strip()

    def _decode_images(self, image_payloads):
        images = []
        for item in image_payloads:
            if item.get("encoding") != "jpeg_base64":
                continue
            raw = base64.b64decode(item.get("data") or "")
            image = self.PILImage.open(io.BytesIO(raw)).convert("RGB")
            images.append(image)
        return images

    def _images_to_frames(self, images):
        # Alpamayo helper expects N_total frames. With one live ROS camera, repeat
        # the latest frame four times to match the training-era temporal format.
        image = images[-1]
        array = self.torch.from_numpy(self.np.array(image)).permute(2, 0, 1)
        frames = []
        for _ in range(self.num_frames_per_camera):
            frames.append(array.clone())
        return self.torch.stack(frames, dim=0)

    @staticmethod
    def _build_question(snapshot):
        compact = {
            "command": snapshot.get("command"),
            "parsed_steps": snapshot.get("parsed_steps"),
            "current_lane": snapshot.get("current_lane"),
            "nav_status": snapshot.get("nav_status"),
            "last_dispatch": snapshot.get("last_dispatch"),
            "detections": snapshot.get("detections"),
            "lane_info": snapshot.get("lane_info"),
            "path": snapshot.get("path"),
            "pose": snapshot.get("pose"),
        }
        return (
            "You are a driving VLA teacher observing a small autonomous track car. "
            "Use the camera image and this ROS navigation snapshot to write one "
            "concise natural-language paragraph. Do not use bullets, headings, "
            "numbered lists, JSON, or labels. Do not issue driving commands. "
            "In 2 to 4 complete sentences, explain whether the parsed intent, "
            "lane choice, target zone, visual lane evidence, and vehicle status "
            "are mutually consistent. If the target zone cannot be identified "
            "visually, say in the same paragraph that map/odom state is required.\n\n"
            f"ROS snapshot:\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
        )


class AlpamayoRealHandler(BaseHTTPRequestHandler):
    runtime = None
    server_version = "NavVLAAlpamayoReal/0.1"

    def log_message(self, fmt, *args):
        print(f"[alpamayo_real_server] {self.address_string()} - {fmt % args}")

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in {"/", "/health"}:
            self._send_json(200, {"ok": True, "service": "alpamayo_real_server"})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/judge":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw or "{}")
            result = self.runtime.judge(payload)
            self._send_json(200, {"ok": True, **result})
        except Exception as exc:
            self._send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-generation-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.98)
    args = parser.parse_args()

    AlpamayoRealHandler.runtime = AlpamayoRuntime(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_generation_length=args.max_generation_length,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    server = ThreadingHTTPServer((args.host, args.port), AlpamayoRealHandler)
    print(f"real Alpamayo endpoint ready: http://{args.host}:{args.port}/judge")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
