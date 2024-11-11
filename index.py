import subprocess
import threading
import webbrowser
import time
import os
import sys

def start_http_server():
    """启动HTTP服务器"""
    print("Starting HTTP server on port 8080...")
    subprocess.Popen([sys.executable, "-m", "http.server", "8080"])

def run_main_script():
    """在py38环境中运行main.py"""
    print("Activating conda environment and running main.py...")
    
    # Windows下激活conda环境的命令
    activate_cmd = "conda activate py38"
    
    # 使用cmd.exe执行conda activate
    process = subprocess.Popen(f"cmd.exe /c {activate_cmd} && python main.py", 
                             shell=True)
    return process

def open_browser():
    """打开浏览器访问指定页面"""
    url = "http://localhost:8080/asan-main.html"
    print(f"Opening browser to {url}")
    webbrowser.open(url)

def main():
    # 1. 启动HTTP服务器
    server_thread = threading.Thread(target=start_http_server)
    server_thread.daemon = True  # 设置为守护线程，这样主程序退出时会自动结束
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(2)
    
    # 2. 运行main.py (在新的conda环境中)
    main_process = run_main_script()
    
    # 等待main.py启动
    time.sleep(3)
    
    # 3. 打开浏览器
    open_browser()
    
    try:
        # 等待main.py执行完成
        main_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        main_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    main()