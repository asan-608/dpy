import sys

def check_python_version():
    # 获取主版本号和次版本号
    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro
    
    # 获取完整的版本信息字符串
    full_version = sys.version
    
    print(f"Python 版本号: {major}.{minor}.{micro}")
    print(f"完整版本信息: {full_version}")

if __name__ == "__main__":
    check_python_version()