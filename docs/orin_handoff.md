# 后续交接：本轮只准备运行发布包

不要修改九动作或回退SFT，不要停止另一个训练进程，不要添加相机/云台。先读 README、migration_inventory、model_contract、validation_report 和 release_status.json。

当前唯一策略 `models/grpo_viewpoint.pt` 为封装577 SFT后第1次GRPO更新权重，依据第2轮实际rollout表现选择。metadata/评测证据在 docs/model_selection.json。模型yaw tokens必须保持映射既有云台pan，所有视角XY/角度执行均禁止身体yaw指令；只有导航可转身体。不要把仿真Xform角度当作真实电机编码器。

先按README `check-models`、`replay --offline` 和38项tests复验。等待已有训练自然结束再独占Isaac，使用 `sim_runtime/run_single_point.py` 的真实控制链，不用Mock取代ARRIVED/STOP。先检查真实资源和Kit MDL渲染，再配有效ARK_API_KEY，验证单点导航→视角→当前帧数字/颜色读取→result.json。必须记录真实角度回读、当前帧ID、动作和失败，接口测试不能当实测。

Orin/真机未适配：先查询JetPack/L4T、aarch64、Python、CUDA、PyTorch、ROS、内存、相机时间戳、两云台关节编码器/速度/限位/故障接口和导航状态协议。该包Isaac仿真运行入口不应部署到真机直接运动。需要独立实机适配和授权验收；不量化、不换网络、不猜测步长/标定、不用0补缺失安全状态。

公共发布前核实 docs/publication.md 的来源许可和私有资产交付；不能擅自上传资源私有目录或API密钥。
