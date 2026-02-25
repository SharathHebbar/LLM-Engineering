import subprocess
import tempfile
import os
import uuid
import json
import textwrap


class DockerSandbox:
    def __init__(self, image="llm-sandbox:latest"):
        self.image = image

    def _build_wrapped_code(self, user_code: str, function_name: str, args: dict):
        """Wraps LLM-generated function code with a safe execution entrypoint"""
        wrapped = f"""
{user_code}

import json
if __name__ == "__main__":
    try:
        result = {function_name}(**{json.dumps(args)})
        print(json.dumps({{"success": True, "result": result}}))
    except Exception as e:
        print(json.dumps({{"success": False, "error": str(e)}}))
"""
        return textwrap.dedent(wrapped)
    
    def run(self, code:str, function_name: str, args: dict, timeout=5):
        wrapped_code = self._build_wrapped_code(code, function_name, args)
        container_name = f"sandbox-{uuid.uuid4().hex}"
        cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network=none",
            "--memory=256m",
            "--cpus=0.5",
            "--pids-limit=64",
            "--read-only",
            "--security-opt=no-new-privileges",
            "--cap-drop=ALL",
            "-i",
            self.image,
            "python", "-c", wrapped_code
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                try:
                    return json.loads(stdout)
                except json.JSONDecodeError:
                    return {"success": False, "error": "Invalid JSON output", "raw_stdout": stdout, "stderr": stderr}
                except Exception as e:
                    return {"success": False, "error": str(e), "raw_stdout": stdout, "stderr": stderr}
            return {
                "success": False,
                "error": "No output produced",
                "stderr": stderr,
            }
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "error": f"Exception occured due to {e}"}