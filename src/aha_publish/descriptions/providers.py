"""Provider client calls for multimodal task description analysis."""

from aha_publish import paths

from .context_blocks import *

def analyze_with_openai(task_paths: TaskInputs, args: argparse.Namespace, text_payload: str, client) -> tuple[dict[str, Any], Usage]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": text_payload,
        }
    ]
    if args.include_grid_image:
        for image_entry in task_paths.image_paths:
            image_data, media_type = load_image_as_base64(image_entry.path)
            content.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_data}",
            })

    response = client.responses.create(
        model=args.openai_model,
        max_output_tokens=args.max_output_tokens,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )
    return parse_json_response(response.output_text), usage_from_openai(response)


def analyze_with_claude(task_paths: TaskInputs, args: argparse.Namespace, text_payload: str, client) -> tuple[dict[str, Any], Usage]:
    content: list[dict[str, Any]] = []
    if args.include_grid_image:
        for image_entry in task_paths.image_paths:
            image_data, media_type = load_image_as_base64(image_entry.path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_data,
                },
            })
    content.append({"type": "text", "text": text_payload})

    # Stream so large max_tokens (big multi-waypoint tasks) don't trip the
    # SDK's non-streaming 10-minute guard.
    with client.messages.stream(
        model=args.claude_model,
        max_tokens=args.max_output_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": content},
        ],
    ) as stream:
        message = stream.get_final_message()
    text = "".join(b.text for b in message.content if b.type == "text")
    return parse_json_response(text), usage_from_claude(message)


def _load_openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def build_client(provider: str):
    if provider == "openai":
        import openai
        api_key = _load_openai_key()
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not found. Set OPENAI_API_KEY."
            )
        return openai.OpenAI(api_key=api_key)
    if provider == "claude":
        import anthropic
        return anthropic.Anthropic()
    raise ValueError(f"Unknown provider '{provider}'. Choose 'openai' or 'claude'.")
