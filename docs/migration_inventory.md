# 发布内容与调用链

此目录是独立的待发布副本，未改动来源仓库、旧模型、数据或进行中的训练。逐文件来源及 SHA256 见 `source_provenance.json`；原仓库存在未提交修改，Git HEAD 仅作辅助证据。

| 原实现 | 发布位置 | 保留/必要修改 |
|---|---|---|
| g1-active-view 的 OfflinePolicy、DINOv2、SIFT | src/g1_active_view | 保持网络、预处理、argmax、历史编码及权重不变 |
| planner/live_single_point.py | src/planner/live_single_point.py | 保留真实导航反馈、50步预算、每5步读数、Stop读数和最多5次单步重试；明确资源根；去显示、探针；禁止超时后历史照片冒充到达 |
| planner/live_grpo_binding.py | src/planner/live_grpo_binding.py | 保留实际云台 Xform 回读；历史 BodyYaw tokens 映射 pan；所有视角速度 tick 的身体 yaw 强制0，包括 XY 伺服修正 |
| planner/live_camera.py、number_color_reader.py、live_reader_worker.py | src/planner | 最新现有传感器 buffer；严格数字/背景色 JSON；独立 API 子进程及超时 |
| body_xy_runtime.py、joint_controller.py | src/gimbal_tvr | 仅保留运行状态与速度 tick/云台控制；不保留启动训练仿真的构造函数 |
| physical_expert_executor / dedicated_sim_validation / shared_relative_executor 等 | src/g1_execution | 提取实际 PolicyExecutor 运行依赖；移除专家采集、目标答案反馈、训练入口；非“规则专家替代模型” |
| unitree_sim_isaaclab/sim_main.py 的单点分支 | sim_runtime/run_single_point.py | 相同 task、DDSRLActionProvider、RobotController、导航和 Planner 对象绑定；显式文件、headless、单实例锁；无旧巡检/重放/遥操作/视频页面 |
| navigation 的 Warp LiDAR、SLAM、ROS bridge、C++ localPlanner 和 inspectionPathFollower | sim_runtime/navigation | 复用原算法/配置/路径库；去未调用 RTX LiDAR 和旧 follower；构建库覆盖改为显式选项；禁止 realRobot=true |
| 已装好云台的 G1 task/presets/控制器 | sim_runtime/tasks、robots、tools | 仅 dex1 wholebody 注册、既有 inspection_camera；不添加额外传感器/安装云台；移除未使用共享图像发布观察项 |
| 原场景、G1 USD、步行 ONNX、目标截图 | resources/private（Git 忽略） | 只在本地复制实际依赖闭包并相对化 USD 资源引用；保存源哈希与副本哈希，不修改源资产 |

没有保留：SFT/GRPO 训练程序、优化器状态、训练轨迹、训练输出、其他权重路线、旧 Planner/多点巡检、视频展示、缓存、运行日志、原 `.git`、真实密钥、内部预建地图。DINOv2仅保留ViT-B/14实际import闭包15个源码文件，内容与已核验源版本逐字节一致，删除不依赖的训练/其他模型模块；完整源树哈希与保留子树哈希都记录，不更新源码版本。

## 信号与动作所有权

1. `scripts/start_mission.py` 读取实际 READY 的 PID/nonce，写 START。
2. `LiveSinglePoint` 根据实时工厂标签几何选择安全导航点，再以测量的 world/SLAM 配准提交 `IsaacNavigationBridge.set_goal`。
3. `arrived()` 必须收到当前 goal_id 的新鲜 `arrival_latched`。给定坐标不是到达证据。
4. `VIEW_REPRODUCING` 调用 `OfflinePolicy.predict(current,target,history)`。导航只更新传感/反馈，`write_commands=False`，不再产生身体转向。
5. 模型 `BodyYawLeft/Right` 仅路由既有云台 pan；pitch 控制既有 tilt；XY 保留身体平移，但禁止身体 yaw 指令。实际身体漂移仍检测，不伪装绝对不动。
6. 实际 Stop 或完成5个视角动作后停止速度、等待新 sensor frame ID/time，再由 API worker 读取这张当前 RGB。每5步可靠读数可直接完成，保持用户此前确认的流程；不会制造策略 Stop。
7. API 返回严格 JSON，保存实际 result.json。模型执行失败不读图；读数失败重试最多5次，每次一步再取新图；认证错误直接失败。

`VIEW_REACHED` 只在真实策略 Stop 且执行成功处标记。API 读数成功与相似度/Stop保持成功不是同一个指标。原“预算耗尽读取历史最佳图”路径已移除。

## 执行局限

云台目前是仿真中既有 kinematic Xform 控制，不是实机电机关节驱动。yaw 重映射及 XY 身体 yaw 禁止是用户要求的执行边界变更，不能继承历史闭环性能；未放宽原执行容差。真实 GPU/Isaac 全链路需待当前训练结束后独占运行验证。
