"""HTTP GUI server and persistence helpers for BT Condition Studio."""

from aha_publish import paths

from .llm_generation import *

def print_conditions(result: dict):
    for stage in result["stages"]:
        print(f"\n{'='*50}")
        print(f"  Stage {stage['stage']} — {stage['name']}")
        print(f"{'='*50}")
        print("  PRECONDITIONS:")
        for c in stage["preconditions"]:
            print(f"    • {condition_text(c)}")
        print("  POSTCONDITIONS:")
        for c in stage["postconditions"]:
            print(f"    • {condition_text(c)}")
        if stage.get("hold_conditions"):
            print("  HOLD CONDITIONS:")
            for c in stage["hold_conditions"]:
                print(f"    • {condition_text(c)}  [{c.get('detector', '')}]")

# ==============================================================================
# 11. WEB GUI SERVER
# ==============================================================================

def same_path(saved_path: str | None, current_path: Path) -> bool:
    if not saved_path:
        return False
    try:
        # saved_path may be repo-relative (portable) or an absolute legacy path.
        return resolve_project_path(Path(saved_path)).resolve() == current_path.resolve()
    except OSError:
        return str(saved_path) == str(current_path)

def load_saved_conditions(args: argparse.Namespace) -> dict | None:
    for path in review_output_candidates(args):
        if not path.exists():
            continue

        try:
            saved_data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        saved_task_path = (
            saved_data.get("waypoint_json_path")
            or saved_data.get("task_context_path")
        )
        if saved_task_path and not same_path(saved_task_path, args.task_context):
            continue

        saved_conditions = saved_review_to_conditions(saved_data)
        if saved_conditions:
            saved_conditions["source_path"] = str(path)
        return saved_conditions

    return None

def save_review_file(args: argparse.Namespace, payload: dict) -> Path:
    args.review_output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "task_name": args.task_name,
        "waypoint_json_path": project_relative_str(args.task_context),
        "task_context_path": project_relative_str(args.task_context),
        "failure_definitions_path": project_relative_str(CONDITION_RULES_PATH),
        "generated": payload.get("generated"),
        "review": payload.get("review"),
    }
    args.review_output.write_text(json.dumps(output, indent=2))
    return args.review_output

def make_gui_handler(args: argparse.Namespace):
    class BTGuiHandler(BaseHTTPRequestHandler):
        def _send_text(self, text: str, status: int = 200, content_type: str = "text/html"):
            encoded = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, data: dict, status: int = 200):
            self._send_text(json.dumps(data), status, "application/json")

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw)

        def _send_error_json(self, error: Exception, status: int = 500):
            self._send_json({"error": str(error)}, status)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send_text(load_design_html())
                return

            if path == "/api/bootstrap":
                try:
                    ensure_input_files(args)
                    task_context = load_task_context_object(args.task_context)
                    failure_definitions = load_failure_definitions()
                    # By default the studio loads the predefined saved BT. Pass
                    # --generate to ignore it and generate a fresh one through the
                    # same generate_conditions pipeline as the terminal output.
                    saved_conditions = (
                        None
                        if getattr(args, "generate", False)
                        else load_saved_conditions(args)
                    )
                    self._send_json(
                        {
                            "provider": args.provider,
                            "task_name": args.task_name,
                            "waypoint_json_path": str(args.task_context),
                            "task_context_path": str(args.task_context),
                            "failure_definitions_path": str(CONDITION_RULES_PATH),
                            "review_output_path": str(args.review_output),
                            "saved_conditions_source_path": (
                                saved_conditions.get("source_path")
                                if saved_conditions
                                else ""
                            ),
                            "auto_generate_on_load": (
                                saved_conditions is None
                                and AUTO_GENERATE_ON_LOAD
                                and not args.no_auto_generate
                            ),
                            "has_saved_conditions": saved_conditions is not None,
                            "saved_conditions": saved_conditions,
                            "overall_description": task_context.get("overall_description", ""),
                            "task_stages": extract_task_stages(task_context),
                            "failure_categories": failure_definition_summaries(failure_definitions),
                            "hold_conditions": hold_condition_detectors(failure_definitions),
                        }
                    )
                except Exception as e:
                    self._send_error_json(e, 400)
                return

            self._send_json({"error": "Not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path

            if path == "/api/generate":
                try:
                    payload = self._read_json()
                    ensure_input_files(args)
                    provider = args.provider.strip().lower()
                    task_context = load_task_context(args.task_context)
                    failure_definitions = load_failure_definitions()
                    client = build_client(provider)
                    result = generate_conditions(
                        failure_definitions,
                        task_context,
                        client,
                        provider,
                        payload.get("guidance", ""),
                        task_context_path=args.task_context,
                        reviewer=getattr(args, "reviewer", None),
                        # The GUI shows Agent 2's reviewed BT verbatim — no cleanup
                        # pipeline and no ground-truth stamping. The cleanup passes
                        # were rewriting Agent 2's fixes (e.g. re-imposing transport
                        # semantics on a stage Agent 2 reclassified as push), so the
                        # browser no longer matched the reviewed output printed in
                        # the terminal. Only hold conditions are still attached, as
                        # those are runtime detectors rather than generated content.
                        postprocess=False,
                    )
                    response = {"result": result}
                    # --generate means "make a fresh BT for this task", so the
                    # generated draft is written straight to the prepared BT
                    # file, overwriting the predefined one. Editing in the GUI
                    # and pressing Save still overwrites it again afterwards.
                    if getattr(args, "generate", False):
                        saved_path = save_review_file(args, {"generated": result})
                        response["saved_path"] = str(saved_path)
                        print(f"Saved generated BT -> {saved_path}")
                    self._send_json(response)
                except Exception as e:
                    self._send_error_json(e, 500)
                return

            if path == "/api/save":
                try:
                    payload = self._read_json()
                    output_path = save_review_file(args, payload)
                    self._send_json({"path": str(output_path)})
                except Exception as e:
                    self._send_error_json(e, 500)
                return

            self._send_json({"error": "Not found"}, 404)

        def log_message(self, format, *args):
            return

    return BTGuiHandler

def run_gui(args: argparse.Namespace) -> None:
    server = ThreadingHTTPServer((args.host, args.port), make_gui_handler(args))
    url = f"http://{args.host}:{args.port}"

    print(f"BT Condition Studio is running at {url}")
    print(f"Provider: {args.provider}")
    print(f"Task context: {args.task_context}")
    if getattr(args, "task_name", None):
        print_task_category(args.task_name)
    print(f"Condition catalogue: {CONDITION_RULES_PATH}")
    print(f"Review output: {args.review_output}")
    if getattr(args, "generate", False):
        print("--generate: the fresh BT will overwrite this file automatically.")
    print("Press Ctrl+C to stop.")

    if AUTO_OPEN_BROWSER and not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BT Condition Studio.")
    finally:
        server.server_close()
