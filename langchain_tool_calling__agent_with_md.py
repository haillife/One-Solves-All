from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from typing import Type, Dict, Any, List, Tuple, Optional
import json
import os
from langchain_core.prompts import (
    ChatPromptTemplate, MessagesPlaceholder,
    HumanMessagePromptTemplate, SystemMessagePromptTemplate
)
import socket
import subprocess
import time
from datetime import datetime
import re
import textwrap
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import create_tool_calling_agent


# ──────────────────────────────────────────────
# 1. Kernel 启动
# ──────────────────────────────────────────────

def start_kernel_if_needed(script_path="python_jupyter_kernel_tool.py", port=5055):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        port_in_use = s.connect_ex(('localhost', port)) == 0

    if port_in_use:
        print(f"⚠️ 端口 {port} 已被占用，尝试终止旧进程...")
        result = subprocess.run(
            f"netstat -ano | findstr :{port}",
            shell=True, capture_output=True, text=True
        )
        pids = set()
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if parts and parts[-1].isdigit():
                pid = int(parts[-1])
                if pid > 4:
                    pids.add(pid)
        for pid in pids:
            print(f"  终止 PID {pid}...")
            subprocess.run(f"taskkill /PID {pid} /F", shell=True)
        time.sleep(2)

    print(f"🚀 正在后台启动内核服务 ({script_path})...")
    try:
        kernel_proc = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"❌ 启动kernel失败: {e}")
        print(f"   script_path={script_path}, cwd={os.getcwd()}")
        return None

    time.sleep(4)
    print("✅ 内核服务已启动。")
    return kernel_proc


# ──────────────────────────────────────────────
# 2. JupyterAPITool
# ──────────────────────────────────────────────

class JupyterAPITool(BaseTool):
    """
    完全兼容 LangChain(BaseTool) + Pydantic 的工程版 Jupyter 工具。
    通过 AGENT_RUN_DIR 环境变量获取图片保存目录。
    """

    name: str = "JupyterAPITool"
    description: str = (
        "Execute Python code inside Jupyter kernel backend "
        "and automatically save matplotlib plots."
    )
    image_save_path: str = "log/default_images"

    _plot_counter: int = PrivateAttr(default=0)

    # ------------------------------------------------------------------
    # Utility: sanitize code
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_code(raw: str) -> str:
        s = raw.strip()
        s = re.sub(r"^\s*```(?:python|py)?\s*\n?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\n?\s*```\s*$", "", s, flags=re.IGNORECASE)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = s.lstrip("\ufeff").replace("\u00A0", " ")
        s = textwrap.dedent(s)
        lines = s.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    @staticmethod
    def _indent_block(block: str, indent: str) -> str:
        return "\n".join((indent + l if l else l) for l in block.splitlines())

    # ------------------------------------------------------------------
    # Filename generator
    # ------------------------------------------------------------------
    def _generate_plot_path(self, save_dir: str) -> str:
        self._plot_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"img_{ts}_{self._plot_counter}.png"
        full_path = os.path.join(save_dir, filename)
        return full_path.replace(os.sep, "/")

    # ------------------------------------------------------------------
    # Inject savefig()
    # ------------------------------------------------------------------
    def _inject_savefig(self, code: str, save_dir: str) -> Tuple[str, List[str]]:
        if "plt.savefig" in code:
            print("📝 Detected explicit plt.savefig() — no injection.")
            return code, []

        plot_paths = []
        pat = re.compile(r"(?m)^(?P<indent>\s*)plt\.show\(\)\s*$")

        def repl(match):
            indent = match.group("indent") or ""
            plot_path = self._generate_plot_path(save_dir)
            plot_paths.append(plot_path)
            savefig_block = textwrap.dedent(
                f"""
                try:
                    import matplotlib.pyplot as plt
                    import json as _json
                    plt.savefig({json.dumps(plot_path)}, dpi=160, bbox_inches='tight')
                    plt.close()
                    print(_json.dumps({{"plot_path": {json.dumps(plot_path)}}}, ensure_ascii=False))
                except Exception as e:
                    print("图片保存失败:", str(e))
                """
            ).rstrip("\n")
            return self._indent_block(savefig_block, indent)

        new_code, n = pat.subn(repl, code)

        if n > 0:
            print(f"📝 replaced {n} plt.show(), generated {len(plot_paths)} images.")

        if n == 0 and ("matplotlib" in code or "plt." in code):
            plot_path = self._generate_plot_path(save_dir)
            plot_paths.append(plot_path)
            auto_block = textwrap.dedent(
                f"""
                try:
                    import matplotlib.pyplot as plt
                    import json as _json
                    plt.savefig({json.dumps(plot_path)}, dpi=160, bbox_inches='tight')
                    plt.close()
                    print(_json.dumps({{"plot_path": {json.dumps(plot_path)}}}, ensure_ascii=False))
                except Exception as e:
                    print("图片保存失败:", str(e))
                """
            ).rstrip("\n")
            new_code = new_code.rstrip() + "\n\n" + auto_block + "\n"
            print(f"📝 Auto-injected savefig → {plot_path}")

        return new_code, plot_paths

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------
    def _run(self, code: str) -> str:
        save_dir = (
            os.getenv("AGENT_RUN_DIR")
            or self.image_save_path
            or "log/default_images"
        )
        os.makedirs(save_dir, exist_ok=True)
        print(f"🗂 Jupyter run_dir = {save_dir}")

        code = self._preprocess_code(code)
        code, plot_paths = self._inject_savefig(code, save_dir)
        for p in plot_paths:
            print(f"📊 Will save image: {p}")

        print("🔹 Sending code to Jupyter API...")
        try:
            resp = requests.post("http://localhost:5055/execute", json={"code": code})
            if resp.status_code != 200:
                return f"❌ Jupyter API Error: {resp.status_code} - {resp.text}"

            outputs = resp.json().get("outputs", [])
            result = ""
            for item in outputs:
                t = item.get("type")
                if t in ("text", "stream"):
                    result += item.get("content", "")
                elif t == "error":
                    result += "\n---ERROR---\n" + item.get("content", "")
            return result.strip()

        except Exception as e:
            return f"❌ Tool Error: {e}"


# ──────────────────────────────────────────────
# 3. LLM & Prompt & Agent
# ──────────────────────────────────────────────

llm_model = "deepseek-ai/DeepSeek-V3.1-Terminus"
siliconflow_api = "https://api.siliconflow.cn/v1"
siliconflow_key = ""


openrouter_api = "https://openrouter.ai/api/v1"
openrouter_key = ""

# llm_model = "openai/gpt-4o-2024-11-20"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    max_tokens=16383,
    disabled_params={"stop": None},
    openai_api_base=siliconflow_api,
    openai_api_key=siliconflow_key
    # openai_api_base=openrouter_api,
    # openai_api_key=openrouter_key
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template('''
You are an expert assistant specialized in side-channel analysis.

I would like you to act as an expert in side-channel analysis and signal processing, helping me analyze vulnerabilities in an encryption system. During the analysis process, please do not ask me any questions; instead, proceed step by step until you provide the final answer.
The file contains a power consumption waveform from an ECC signing process, where each segment represents an operation. There are 288 operations in this waveform. Please identify these operations, which can be classified into two types: D and A.

To complete this task, you may need to follow these steps:

**Step 1 — Segmentation**
Segment the waveform into exactly 288 equal-length parts. Each part represents one operation.

**Step 2 — Dimensionality Reduction and Clustering**
Apply dimensionality reduction first — PCA is recommended as a starting point. Then apply clustering — KMeans with k=2 is recommended. Assign the larger cluster as D (since D operations are more frequent).

**Step 3 — Constraint Verification**
After every classification attempt, verify all three of the following and print the results:
1. Exactly 192 D operations
2. First operation is D
3. No two consecutive A operations (there must be at least one D operation between any two A operations)

**Step 4 — If Constraints Are Not Satisfied**
Do NOT modify labels manually to force the count. Instead, adjust the analysis method:
- First, try adjusting the PCA parameters (e.g., change the number of components to retain) and re-run clustering.
- If that still fails, replace PCA with a different dimensionality reduction method (e.g., t-SNE, UMAP, or kernel PCA), then re-cluster.
- If KMeans continues to fail, try a different clustering algorithm (e.g., DBSCAN, GMM, or agglomerative clustering).
- Keep iterating with different combinations until all three constraints are satisfied.

Print the D count and pass/fail for each constraint after every attempt.

**MANDATORY CONSTRAINTS - VIOLATION IS NOT ALLOWED:**
1. You MUST use JupyterAPITool to execute ALL analysis. You are FORBIDDEN from assuming, estimating, or fabricating any results without running actual code.
2. Classification labels MUST come from clustering output only. Do NOT write any code that flips or reassigns labels based on target counts after clustering — this includes any logic that iterates through the sequence to force D count to 192. This is FORBIDDEN and counts as a fabricated result.
3. You MUST NOT give a Final Answer until an Observation confirms all three constraints are satisfied simultaneously.
4. If constraints are not met, change the feature extraction strategy or clustering parameters and retry. Do not give up after one or two attempts.
    '''),
    HumanMessagePromptTemplate.from_template("{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template('''
# You are an expert assistant specialized in side-channel analysis.
#
# I would like you to act as an expert in side-channel analysis and signal processing, helping me analyze vulnerabilities in an encryption system. During the analysis process, please do not ask me any questions; instead, proceed step by step until you provide the final answer.
# The file contains a power consumption waveform from an ECC signing process, where each segment represents an operation. There are 256 operations in this waveform. Please identify these operations, which can be classified into two types: D and A.
#
# To complete this task, you may need to follow these steps:
#
# **Step 1 — Segmentation**
# Segment the waveform into exactly 256 equal-length parts. Each part represents one operation.
#
# **Step 2 — Dimensionality Reduction and Clustering**
# Apply dimensionality reduction first — PCA is recommended as a starting point. Then apply clustering — KMeans with k=2 is recommended. Assign the larger cluster as D (since D operations are more frequent).
#
# **MANDATORY CONSTRAINTS - VIOLATION IS NOT ALLOWED:**
# 1. You MUST use JupyterAPITool to execute ALL analysis. You are FORBIDDEN from assuming, estimating, or fabricating any results without running actual code.
# 2. Classification labels MUST come from clustering output only.
#
#     '''),
#     HumanMessagePromptTemplate.from_template("{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])





tools = [JupyterAPITool()]

agent = create_tool_calling_agent(llm, tools, prompt)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True,
#     allow_dangerous_code=True,
#     return_intermediate_steps=True,
#     max_iterations=100,
#     early_stopping_method="generate",
# )


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
    return_intermediate_steps=True,
    max_iterations=100,
    max_execution_time=600,          # 最多运行600秒（10分钟）
    early_stopping_method="force",   # 超限时直接停止而不是强制生成答案
)


# ──────────────────────────────────────────────
# 4. Prompt 预览（调试用）
# ──────────────────────────────────────────────

def inspect_prompt(query: str, prompt) -> Optional[str]:
    """
    预览某个 query 会生成什么 prompt，不执行 agent。
    """
    try:
        # tool_calling_agent 的 prompt 只需要 input 和 agent_scratchpad
        prompt_inputs = {
            "input": query,
            "agent_scratchpad": [],  # tool_calling_agent 用 list
        }
        formatted_prompt = prompt.format_prompt(**prompt_inputs).to_string()
        print("\n🧠 ===== Prompt Preview Start =====\n")
        print(formatted_prompt)
        print("\n🧠 ===== Prompt Preview End =====\n")
        return formatted_prompt
    except Exception as e:
        print(f"❌ 生成 prompt 失败: {e}")
        return None


# ──────────────────────────────────────────────
# 5. Agent 执行 + Markdown 日志
# ──────────────────────────────────────────────

def run_agent_and_log_md(
    agent_executor: AgentExecutor,
    query: str,
    llm,
    prompt_text: Optional[str],
    log_path: str = "log/agent_log_PLACEHOLDER/agent_log.md"
):
    # ── 5-1. Markdown 日志回调 ──────────────────────────────────────────
    class MarkdownLogHandler(BaseCallbackHandler):
        def __init__(self, run_dir: str):
            self.records: List[str] = []
            self.run_dir = os.path.abspath(run_dir)
            self.step_idx: int = 0
            self.pending_action: Optional[dict] = None

        # ── helpers ────────────────────────────────────────────────────
        def _safe_json_load(self, text):
            try:
                if isinstance(text, (dict, list)):
                    return text
                return json.loads(str(text).strip())
            except Exception:
                pass
            try:
                clean = str(text)
                start = clean.find("{")
                if start >= 0:
                    obj, _ = json.JSONDecoder().raw_decode(clean[start:])
                    return obj
            except Exception:
                pass
            return None

        def _collect_image_paths(self, obj):
            paths = []
            if isinstance(obj, dict):
                for v in obj.values():
                    paths.extend(self._collect_image_paths(v))
            elif isinstance(obj, list):
                for v in obj:
                    paths.extend(self._collect_image_paths(v))
            elif isinstance(obj, str) and obj.lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                paths.append(obj)
            return paths

        def _extract_image_paths_from_text(self, text: str) -> List[str]:
            patterns = [
                r'"plot_path"\s*:\s*"([^"]+)"',
                r'"image_path"\s*:\s*"([^"]+)"',
                r'"img_path"\s*:\s*"([^"]+)"',
                r'"file_path"\s*:\s*"([^"]+)"',
                r'([\w./\\-]+\.(?:png|jpg|jpeg|svg))',
            ]
            found = []
            for pat in patterns:
                found.extend(re.findall(pat, text))
            return found

        def _filter_paths_to_run_dir(self, paths: List[str]) -> List[str]:
            kept = []
            for p in paths:
                p_str = str(p).replace("\\", "/")
                if not p_str:
                    continue
                basename = os.path.basename(p_str)
                if os.path.exists(p_str) or os.path.exists(os.path.join(self.run_dir, basename)):
                    kept.append(basename)
            seen, uniq = set(), []
            for b in kept:
                if b not in seen:
                    uniq.append(b)
                    seen.add(b)
            return uniq

        def _flush_pending(self):
            """把 pending action 写入并清空。"""
            if self.pending_action:
                self.records.append(self.pending_action["content"])
                self.records.append("\n**Observation:**\n_[no output from tool]_\n")
                self.pending_action = None

        # ── callbacks ──────────────────────────────────────────────────
        def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
            self._flush_pending()
            self.step_idx += 1

            parts = ["\n---\n", f"**Step {self.step_idx}**\n"]

            # tool_calling_agent 的 action.log 通常为空，直接记录工具名和参数
            parts.append(f"\n**Action:** `{action.tool}`\n")

            raw_input = action.tool_input
            if isinstance(raw_input, dict):
                # tool_calling_agent 传的是 dict（{"code": "..."}）
                clean_input = raw_input.get("code", json.dumps(raw_input, ensure_ascii=False))
            else:
                clean_input = re.sub(r'^```\w*\n?|```$', "", str(raw_input).strip(), flags=re.MULTILINE)

            parts.append(f"\n**Action Input:**\n```python\n{clean_input}\n```\n")

            self.pending_action = {"step": self.step_idx, "content": "".join(parts)}

        def on_tool_end(self, output: str, **kwargs: Any) -> Any:
            content = "" if output is None else str(output).strip()

            if self.pending_action:
                self.records.append(self.pending_action["content"])
                self.pending_action = None

            self.records.append("\n**Observation:**\n")
            self.records.append(f"{content}\n" if content else "_[no output from tool]_\n")

            # 图片收集
            image_paths: List[str] = []
            json_data = self._safe_json_load(content)
            if json_data:
                image_paths.extend(self._collect_image_paths(json_data))
            image_paths.extend(self._extract_image_paths_from_text(content))

            if image_paths:
                cleaned = self._filter_paths_to_run_dir(image_paths)
                if cleaned:
                    self.records.append("\n**Generated Images:**\n")
                    for fname in cleaned:
                        self.records.append(
                            f"- {fname}\n"
                            f'<img src="./{fname}" alt="{fname}" '
                            f'style="max-width: 100%; border: 1px solid #ddd; margin: 6px 0;" />\n'
                        )
                    self.records.append("\n")

        def on_chain_end(self, outputs, **kwargs: Any) -> Any:
            self._flush_pending()

    # ── 5-2. 执行主体 ───────────────────────────────────────────────────
    try:
        if not log_path:
            raise ValueError("log_path 不能为空")

        if log_path.lower().endswith(".md"):
            run_dir = os.path.dirname(log_path) or "."
        else:
            run_dir = log_path
            log_path = os.path.join(run_dir, "agent_log.md")

        os.makedirs(run_dir, exist_ok=True)
        os.environ["AGENT_RUN_DIR"] = run_dir  # JupyterAPITool 读取此变量

        # 同步 tool 的保存目录
        for tool in agent_executor.tools:
            if hasattr(tool, 'image_save_path'):
                tool.image_save_path = run_dir

        start_kernel_if_needed()

        start_time = time.time()
        log_handler = MarkdownLogHandler(run_dir)

        print(f"⏳ Agent Running... Logs: {log_path}")
        print(f"📂 Run Directory: {run_dir}")

        # ★ 修复：invoke 只传 prompt 里实际存在的变量
        result = agent_executor.invoke(
            {"input": query},
            config={"callbacks": [log_handler]}
        )

        duration = time.time() - start_time
        execution_log_body = "".join(log_handler.records)

        # ── 组装 Markdown ────────────────────────────────────────────
        # ★ 修复：用 llm.model 取模型名
        model_name = getattr(llm, 'model', getattr(llm, 'model_name', 'N/A'))

        header = f"# Agent Execution Log\n**Query:**\n```text\n{query}\n```\n"

        summary = (
            f"\n---\n## Execution Summary\n"
            f"- **Model:** `{model_name}`\n"
            f"- **Time:** `{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"- **Duration:** `{duration:.2f}s`\n"
            f"- **Path:** `{run_dir}`\n"
        )

        prompt_section = ""
        if prompt_text:
            prompt_section = (
                f"\n---\n## Prompt\n"
                f"<details><summary>Expand</summary>\n\n"
                f"```\n{prompt_text}\n```\n\n</details>\n"
            )

        log_section = f"\n---\n## Execution Log\n{execution_log_body}\n"
        output_text = result.get("output", "").strip()
        final_section = f"\n---\n**Final Output:**\n```text\n{output_text}\n```"

        final_content = header + summary + prompt_section + log_section + final_section

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"✅ Log saved: {log_path}")
        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        if 'log_handler' in locals() and log_path:
            try:
                partial = "".join(log_handler.records)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"# Failed Log\n{partial}\n\nERROR: {e}")
            except Exception:
                pass
        return None


if __name__ == "__main__":

    query = (
        "Perform Simple Power Analysis (SPA) on the ECC algorithm trace collected from a single-chip microcomputer. The trace file is located at: ./trace2/Simulated-1-20.npy, and the trace can be divided into 1536 segments. After classification, output the complete operation sequence to a file named ./trace2/Simulated-1-20-result.txt, with all operations written on a single line with no spaces (e.g., DADADADADDDD...)."
    )

    # ★ 修复：inspect_prompt 只传 prompt，不传多余变量
    prompt_text = inspect_prompt(query, prompt)

    # custom_tag = "Kyber-_GPT4o1120"
    custom_tag = "Kyber-_DeepseekV31T"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = f"log/agent_log_{custom_tag}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    result = run_agent_and_log_md(
        agent_executor,
        query,
        llm=llm,
        prompt_text=prompt_text,
        log_path=run_dir
    )

