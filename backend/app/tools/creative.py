"""Creative tools: the chat model delegates images, edits, video and speech to other models.

Each tool returns text for the model plus ``attachments`` the UI shows inline (an image, a
video job that turns into a clip, an audio player). Results include ids, so follow-ups like
"make it nighttime" can refer back to the right image.
"""

import math

from app import delegation, studio
from app.studio import StudioError
from app.tools import INT, STR, Tool, ToolError, _obj

ASPECTS = {
    "square": (1, 1),
    "portrait": (3, 4),
    "landscape": (4, 3),
    "wide": (16, 9),
    "tall": (9, 16),
}
INSTALL_HINT = "Tell the user they can install one from the Models page."


def dims(aspect: str, base_w: int, base_h: int, multiple: int) -> tuple[int, int]:
    """Keep the model's native pixel count for the requested aspect ratio."""
    w, h = ASPECTS.get(aspect, (1, 1))
    scale = math.sqrt(base_w * base_h / (w * h))
    snap = lambda x: max(256, round(x * scale / multiple) * multiple)  # noqa: E731
    return snap(w), snap(h)


def _pick(app, capability: str, requested: str | None, what: str) -> dict:
    model = delegation.choose(app, capability, requested)
    if model is None:
        raise ToolError(f"No {what} model is installed. {INSTALL_HINT}")
    return model


def creative_tools(app) -> list[Tool]:
    """Tools for every kind of media that has an installed model."""
    tools: list[Tool] = []

    async def generate_image(args: dict) -> dict:
        model = _pick(app, "image", args.get("model"), "image")
        _, variant = studio.resolve(app, model["id"], "image", "image")
        base = variant["engine_config"].get("size") or 1024
        width, height = dims(str(args.get("aspect") or "square"), base, base, 64)
        try:
            r = await studio.create_image(
                app,
                model["id"],
                str(args["prompt"]),
                negative_prompt=args.get("negative_prompt"),
                width=width,
                height=height,
                seed=args.get("seed"),
                source="chat",
            )
        except StudioError as exc:
            raise ToolError(str(exc)) from exc
        finally:
            await studio.park_media(app)
        image_id = r["ids"][0]
        return {
            "content": (
                f"Created image {image_id} with {model['name']} ({width}x{height}, seed {r['seed']}). "
                f'The user can see it. To change it, call edit_image with image_id="{image_id}".'
            ),
            "attachments": [{"type": "image", "id": image_id}],
        }

    async def edit_image(args: dict) -> dict:
        image_id = str(args["image_id"]).strip()
        instruction = str(args["instruction"])
        record = studio.image_record(app, image_id)
        if record is None:
            raise ToolError(f"Image {image_id} not found. Use the id from an earlier tool result.")
        model = _pick(app, "image", args.get("model"), "image")
        params = record.get("params") or {}
        width, height = params.get("width"), params.get("height")
        try:
            if await studio.supports_editing(app, model["id"]):
                r = await studio.create_image(
                    app,
                    model["id"],
                    instruction,
                    width=width,
                    height=height,
                    init_image_id=image_id,
                    source="chat",
                )
                how = f"Edited image {image_id} with {model['name']}"
            elif record["model"] == "upload":
                raise ToolError(
                    f"{model['name']} can't edit existing images. An image model that supports editing "
                    f"(e.g. FLUX.2 [klein]) is needed. {INSTALL_HINT}"
                )
            else:
                # Can't edit pixels: regenerate with the adjusted description and the same seed.
                prompt = f"{record['prompt']}. {instruction}"
                r = await studio.create_image(
                    app,
                    model["id"],
                    prompt,
                    width=width,
                    height=height,
                    seed=params.get("seed"),
                    source="chat",
                )
                how = (
                    f"{model['name']} can't edit images directly, so I regenerated {image_id} with an "
                    "adjusted prompt and the same seed (the composition may change)"
                )
        except StudioError as exc:
            raise ToolError(str(exc)) from exc
        finally:
            await studio.park_media(app)
        new_id = r["ids"][0]
        return {
            "content": f'{how}. New image: {new_id} (the user can see it). Edit it further with image_id="{new_id}".',
            "attachments": [{"type": "image", "id": new_id}],
        }

    async def generate_video(args: dict) -> dict:
        model = _pick(app, "video", args.get("model"), "video")
        _, variant = studio.resolve(app, model["id"], "image", "video")
        cfg = variant["engine_config"]
        width, height = dims(
            str(args.get("aspect") or "landscape"),
            cfg.get("width", 832),
            cfg.get("height", 480),
            32,
        )
        seconds = min(10, max(1, int(args.get("seconds") or 2)))
        fps = cfg.get("fps", 16)
        try:
            job = studio.start_video(
                app,
                model["id"],
                str(args["prompt"]),
                negative_prompt=args.get("negative_prompt"),
                width=width,
                height=height,
                frames=seconds * fps + 1,
                fps=fps,
            )
        except StudioError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "content": (
                f"Started a {seconds}s video with {model['name']} (job {job['id']}). It renders in the "
                "background and takes several minutes; it will appear in the chat when done. Don't "
                "wait for it: tell the user it's on its way."
            ),
            "attachments": [{"type": "video", "job_id": job["id"]}],
        }

    async def speak(args: dict) -> dict:
        model = _pick(app, "speech", args.get("model"), "text-to-speech")
        try:
            rec = await studio.speak_to_library(
                app, model["id"], str(args["text"]), args.get("voice")
            )
        except StudioError as exc:
            raise ToolError(str(exc)) from exc
        return {
            "content": f"Spoke the text with {model['name']}. The user can play it.",
            "attachments": [{"type": "audio", "id": rec["id"]}],
        }

    if delegation.installed_for(app, "image"):
        tools += [
            Tool(
                "generate_image",
                "Create an image from a text description. Write a vivid, detailed prompt (subject, "
                "setting, style, lighting, composition) rather than repeating the user's words.",
                _obj(
                    ["prompt"],
                    prompt=STR,
                    aspect=STR | {"enum": list(ASPECTS)},
                    negative_prompt=STR
                    | {"description": "Things to avoid (some models ignore it)"},
                    model=STR | {"description": "Only if the user asks for a specific model"},
                    seed=INT,
                ),
                generate_image,
                summary_arg="prompt",
            ),
            Tool(
                "edit_image",
                "Change an existing image (one you created, or one the user attached), e.g. 'make it "
                "nighttime', 'add a red hat'. Pass the image id and a short instruction.",
                _obj(["image_id", "instruction"], image_id=STR, instruction=STR, model=STR),
                edit_image,
                summary_arg="instruction",
            ),
        ]
    if delegation.installed_for(app, "video"):
        tools.append(
            Tool(
                "generate_video",
                "Create a short video clip (a few seconds) from a text description. Slow: renders in "
                "the background for several minutes.",
                _obj(
                    ["prompt"],
                    prompt=STR,
                    seconds=INT | {"description": "1-10, default 2"},
                    aspect=STR | {"enum": ["landscape", "portrait", "square", "wide", "tall"]},
                    negative_prompt=STR,
                    model=STR,
                ),
                generate_video,
                summary_arg="prompt",
            )
        )
    if delegation.installed_for(app, "speech"):
        tools.append(
            Tool(
                "speak",
                "Read text aloud as audio the user can play (text-to-speech).",
                _obj(
                    ["text"],
                    text=STR,
                    voice=STR | {"description": "e.g. af_heart, am_adam, bf_emma"},
                ),
                speak,
                summary_arg="text",
            )
        )
    return tools


CREATIVE_PROMPT = """You can create media by delegating to specialist models through tools:
{tools}
When the user asks for media, call the tool (don't just describe what you would make). Write
rich, specific prompts. For changes to an image you made or the user attached ("make it
nighttime", "remove the car"), call edit_image with that image's id from the earlier tool
result or attachment note; don't start over unless asked. After a tool runs, reply briefly;
the user already sees the result."""


def creative_prompt(tools: list[Tool]) -> str | None:
    names = {
        "generate_image": "- generate_image / edit_image: create and change images",
        "generate_video": "- generate_video: short video clips (slow; renders in the background)",
        "speak": "- speak: read text aloud",
    }
    lines = [names[t.name] for t in tools if t.name in names]
    return CREATIVE_PROMPT.format(tools="\n".join(lines)) if lines else None
