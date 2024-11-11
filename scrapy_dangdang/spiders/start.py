import os
import subprocess
import sys

def run_scrapy_with_conda():
    try:
        # 获取脚本所在的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建spiders目录的路径
        spiders_dir = os.path.join(script_dir, 'spiders')
        
        # 如果当前已经在spiders目录，就使用当前目录
        if os.path.basename(script_dir) == 'spiders':
            working_dir = script_dir
        else:
            working_dir = spiders_dir
            
        # 确保目录存在
        if not os.path.exists(working_dir):
            print(f"Error: 目录找不到哦啊: {working_dir}")
            return False
            
        print(f"执行: {working_dir}")
        
        # 构建运行爬虫的命令
        command = (
            'cmd /c '
            'call activate py38 && '
            f'cd /d "{working_dir}" && '
            'scrapy crawl dang'
        )
        
        print("启动...")
        
        # 使用subprocess.Popen来执行命令
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ),
            universal_newlines=True
        )
        
        # 实时输出结果
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            
            if output:
                print(output.strip())
            if error:
                print(error.strip(), file=sys.stderr)
            
            # 检查进程是否结束
            if output == '' and error == '' and process.poll() is not None:
                break
        
        # 获取返回码
        return_code = process.poll()
        
        if return_code == 0:
            print("\n成功!")
            return True
        else:
            print(f"\n错误: {return_code}")
            return False
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return False

if __name__ == "__main__":
    run_scrapy_with_conda()