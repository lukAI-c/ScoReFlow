"""
单独下载预训练权重的脚本

使用方法:
    python script/download_checkpoint.py --path "pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_42/checkpoint/state_1500.pt"

或者直接在代码中修改 checkpoint_path 变量
"""

import os
import sys
import gdown
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from script.download_url import get_checkpoint_download_url


def download_checkpoint(checkpoint_path: str, force_download: bool = False):
    """
    下载指定的预训练权重
    
    Args:
        checkpoint_path: 权重文件的相对路径（相对于项目根目录）
        force_download: 是否强制重新下载（即使文件已存在）
    """
    # 转换为绝对路径
    abs_path = PROJECT_ROOT / checkpoint_path
    
    # 检查文件是否已存在
    if abs_path.exists() and not force_download:
        print(f"✅ 权重文件已存在: {abs_path}")
        print(f"   如需重新下载，请使用 --force 参数")
        return
    
    # 创建一个临时配置对象
    class TempConfig:
        def __init__(self, path):
            self.base_policy_path = path
        
        def get(self, key, default=None):
            return default
    
    cfg = TempConfig(checkpoint_path)
    
    # 获取下载链接
    try:
        download_url = get_checkpoint_download_url(cfg)
    except Exception as e:
        print(f"❌ 无法找到该路径对应的下载链接: {checkpoint_path}")
        print(f"   错误信息: {e}")
        print(f"\n💡 提示: 请检查路径是否正确，或查看 script/download_url.py 中支持的路径列表")
        return
    
    if not download_url or download_url == "":
        print(f"❌ 该权重文件暂无下载链接: {checkpoint_path}")
        print(f"   可能需要您自己训练或等待官方提供")
        return
    
    # 创建目标目录
    dir_name = abs_path.parent
    if not dir_name.exists():
        print(f"📁 创建目录: {dir_name}")
        dir_name.mkdir(parents=True, exist_ok=True)
    
    # 下载文件
    print(f"📥 开始下载...")
    print(f"   来源: {download_url}")
    print(f"   目标: {abs_path}")
    
    try:
        gdown.download(url=download_url, output=str(abs_path), fuzzy=True)
        print(f"✅ 下载成功!")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print(f"\n💡 提示: 您也可以手动从以下链接下载:")
        print(f"   {download_url}")


def list_available_checkpoints():
    """列出一些常用的可下载权重路径"""
    print("\n📋 常用的预训练权重路径示例:\n")
    
    examples = [
        ("Hopper (ReFlow)", "pretrain/hopper-v2/ReFlow/2025-02-06_01-35-03_42/checkpoint/state_1500.pt"),
        ("Hopper (ShortCut)", "pretrain/hopper-medium-v2_pre_shortcut_mlp_ta4_td20/2025-04-25_08-57-19_42/state_40.pt"),
        ("Walker2d (ReFlow)", "pretrain/walker2d-v2/ReFlow/2025-02-06_01-39-14_42/checkpoint/state_1500.pt"),
        ("Walker2d (ShortCut)", "pretrain/walker2d-v2/ShortCut/2025-04-25_12-55-43_42/checkpoint/state_40.pt"),
        ("Ant (ReFlow)", "pretrain/ant-v2/ReFlow/2025-02-06_01-39-14_42/checkpoint/state_1500.pt"),
        ("Humanoid (DDPM)", "pretrain/Humanoid-medium-v3_pre_diffusion_mlp_ta4_td20/2025-05-01_19-04-33_42/checkpoint/state_40.pt"),
        ("Humanoid (ReFlow)", "pretrain/Humanoid-v3_pre_reflow_mlp_ta4_td20/2025-05-01_18-18-08_42/checkpoint/state_50.pt"),
        ("Kitchen Complete", "pretrain/kitchen-complete-v0_pre_shortcut_mlp_ta4_td4/2024-07-10_01-50-52/checkpoint/state_5000.pt"),
    ]
    
    for i, (name, path) in enumerate(examples, 1):
        print(f"{i}. {name}")
        print(f"   {path}\n")
    
    print("💡 更多路径请查看: script/download_url.py 中的 get_checkpoint_download_url() 函数\n")


def main():
    parser = argparse.ArgumentParser(description="下载 ReinFlow 预训练权重")
    parser.add_argument(
        "--path",
        type=str,
        help="权重文件的相对路径（相对于项目根目录）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使文件已存在）"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出常用的可下载权重路径"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_checkpoints()
        return
    
    if not args.path:
        print("❌ 请指定要下载的权重路径!")
        print("\n使用方法:")
        print('  python script/download_checkpoint.py --path "pretrain/hopper-v2/ReFlow/.../state_1500.pt"')
        print("\n或者查看可用路径:")
        print("  python script/download_checkpoint.py --list")
        return
    
    download_checkpoint(args.path, args.force)


if __name__ == "__main__":
    main()

