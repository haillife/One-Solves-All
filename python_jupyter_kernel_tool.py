# 文件名: python_jupyter_kernel_tool.py

from flask import Flask, request, jsonify
from jupyter_client import KernelManager
import sys
import os
import atexit
import signal

# ==================== 终极、根本性的修复 ====================
import matplotlib
matplotlib.use('Agg')

# ==================== Matplotlib 中文字体修复 ====================
import warnings
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# ==========================================================

app = Flask(__name__)

km = None
kc = None

def cleanup_kernel():
    global km
    if km:
        print("🛑 正在清理 Jupyter Kernel 进程...", flush=True)
        km.shutdown_kernel(now=True)
        print("✅ Kernel 已关闭。", flush=True)

atexit.register(cleanup_kernel)

def signal_handler(sig, frame):
    print('\n收到退出信号，正在关闭服务...', flush=True)
    cleanup_kernel()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==================== 启动 Kernel ====================
print(f"🚀 初始化 Jupyter Kernel Manager...", flush=True)
km = KernelManager()
km.start_kernel()
kc = km.client()
kc.start_channels()
print("✅ Jupyter Kernel 已就绪。", flush=True)

# ★ 修复：执行初始化代码并等待完成，防止残留消息污染后续队列
init_msg_id = kc.execute("""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
import matplotlib.pyplot as plt
""")

# 等待初始化的 idle 消息，彻底清空队列
print("⏳ 等待 Kernel 初始化完成...", flush=True)
while True:
    try:
        msg = kc.get_iopub_msg(timeout=15)
        if (msg['header']['msg_type'] == 'status' and
                msg['content'].get('execution_state') == 'idle' and
                msg.get('parent_header', {}).get('msg_id') == init_msg_id):
            break
    except Exception:
        break
print("✅ Kernel 初始化消息已清空，队列干净。", flush=True)


@app.route('/execute', methods=['POST'])
def execute_code():
    data = request.json
    code = data.get("code", "")

    if not code:
        return jsonify({"error": "No code provided"}), 400

    try:
        print("\n===================== 收到代码执行请求 =====================", flush=True)
        print(code, flush=True)
        print("============================================================\n", flush=True)

        # ★ 修复：drain超时从0.05改为0.5，避免残留消息没清干净
        drained = 0
        while True:
            try:
                kc.get_iopub_msg(timeout=0.5)
                drained += 1
            except:
                break
        if drained > 0:
            print(f"⚠️ 清理了 {drained} 条残留消息", flush=True)

        msg_id = kc.execute(code)
        print(f"📤 执行请求已发送, msg_id={msg_id}", flush=True)

        outputs = []
        idle_received = False

        while not idle_received:
            try:
                msg = kc.get_iopub_msg(timeout=300)

                parent_msg_id = msg.get('parent_header', {}).get('msg_id')
                if parent_msg_id != msg_id:
                    print(f"⏭️ 跳过不匹配的消息: {msg['header']['msg_type']}", flush=True)
                    continue

                msg_type = msg['header']['msg_type']
                content = msg.get('content', {})

                if msg_type == 'status':
                    if content.get('execution_state') == 'idle':
                        idle_received = True
                        print("📥 收到 idle 状态，执行完成", flush=True)
                    continue

                if msg_type == 'stream':
                    text = content.get('text', '')
                    outputs.append({'type': 'stream', 'content': text})
                    print(f"[Stream]: {text.strip()}", flush=True)

                elif msg_type in ('display_data', 'execute_result'):
                    data_dict = content.get('data', {})
                    if 'image/png' in data_dict:
                        outputs.append({'type': 'image/png', 'content': data_dict['image/png']})
                        print("[Image Generated]", flush=True)
                    elif 'text/plain' in data_dict:
                        text = data_dict['text/plain']
                        outputs.append({'type': 'text', 'content': text})
                        print(f"[Result]: {text[:100]}...", flush=True)

                elif msg_type == 'error':
                    traceback_content = "\n".join(content.get('traceback', []))
                    outputs.append({'type': 'error', 'content': traceback_content})
                    print(f"[Error]: {traceback_content[:200]}...", flush=True)

            except Exception as e:
                print(f"⚠️ 消息接收异常: {e}", flush=True)
                break

        if not outputs:
            outputs.append({'type': 'text', 'content': '(代码执行完成，无输出)'})

        print(f"✅ 代码执行完成，返回 {len(outputs)} 条输出。", flush=True)
        return jsonify({"outputs": outputs})

    except Exception as e:
        print(f"❌ Server Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500


if __name__ == '__main__':
    print("🌐 正在启动 Flask 服务 (Port: 5055)...", flush=True)
    app.run(host="127.0.0.1", port=5055, debug=False, threaded=True, use_reloader=False)