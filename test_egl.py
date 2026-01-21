import os
# 强制指定渲染后端
os.environ["MUJOCO_GL"] = "egl"

# 如果你之前发现需要 LD_PRELOAD，确保在运行此脚本前 export 了它
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1

try:
    from mujoco_py import MjRenderContextOffscreen, MjSim, load_model_from_xml
    # 修正后的 XML 格式
    xml_str = """
    <mujoco>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        </worldbody>
    </mujoco>
    """
    model = load_model_from_xml(xml_str)
    sim = MjSim(model)
    
    # 这里是核心：如果能挺过这一行不报错，EGL 就彻底修复了
    ctx = MjRenderContextOffscreen(sim, device_id=0)
    
    print("✅ 完美！EGL 渲染上下文创建成功。")
    print("这说明你的驱动挂载路径和环境变量已经避开了 Conda 的干扰。")
    
except Exception as e:
    print(f"❌ 依然失败: {e}")