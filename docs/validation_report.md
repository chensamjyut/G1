# 实际验证报告

日期：2026-11-03。范围：独立发布副本。未停止或更改正在运行的0–2m训练；未启动新Isaac窗口或额外仿真实例；未执行真实机器人控制。

## A：已通过的模型与接口验证

- 选定 GRPO真实checkpoint加载成功；完整9动作表与 `visual_stop_residual` 网络严格匹配；metadata 的SFT祖先hash与封装577权重实际hash一致。
- 原 `MonotoneInference` 直接生成预期结果，不使用部署包生成自己的预期。source SHA、输入SHA、历史、seed、eval/no_grad记录在fixture和真实图像回归证据。
- 真实图像对：相同原环境PyTorch2.13 CPU，原实现与部署实现 **logits/probabilities/similarity误差均0**，动作MoveXPos、91内点一致。
- 全新CPU环境PyTorch2.8：真实图像最大logits绝对误差 **4.76837158203125e-6**，概率 **5.960464477539063e-8**，相似度 **1.1920928955078125e-7**，动作/内点一致。
- 合成4组基准：有匹配位移、相同图Stop、同黑图、不同无纹理图全部通过；最大logits误差2.0742416381835938e-5。容差预设为FP32跨版本logits1e-4、概率1e-5、sim1e-5，动作/内点严格一致，未按失败结果放宽。
- 断网受控：禁止Python socket connect/sendto/getaddrinfo，仍实际完成DINOv2/SIFT推理。没有使用torch hub缓存。
- 38项接口/单元测试通过，见`unit_tests.xml`；包含历史最近8步/reset、yaw转发仅云台、XY伺服身体yaw指令0、ARRIVED关联goal/freshness、步数预算失败、不读历史最佳帧、相机帧新鲜度、周期读数/真实Stop分离、认证错误/解析失败、模型缺失/hash/LFS指针拒绝。带double的测试均明确不代表真实物理执行。

## B：独立安装和组件交付

- 新 `/tmp/g1-release-venv`：Python3.11.15，`include-system-site-packages=false`。CPU PyTorch2.8.0+cpu；numpy1.26.4、OpenCV4.11.0.86、Pillow11.2.1、PyYAML6.0.2，完整版本见requirements。
- 实际安装：`python -m pip install -e '.[test,reader]'`。从仓库外 `/tmp` 且无原项目PYTHONPATH执行check-models、environment、replay和pytest。
- wheel已从全新本地Git克隆实际构建并安装（替换editable安装），从仓库外、空TORCH_HOME、无原工程sys.path通过4组离线回放及38项测试。两个LFS权重在clone中校验通过，未复制未跟踪文件或私有资源。检查结果另见`delivery_validation.json`，不将本地克隆写成GitHub下载验证。
- 两个模型的SHA256及DINOv2树哈希实际校验通过。所有大权重标记Git LFS，非普通Git二进制。
- C++导航：首次默认构建失败，原因本机GCC11链接桩缺少ROS所需 `GLIBCXX_3.4.30`；显式指定当前系统已有libstdc++.so.6后，两包构建成功。未改系统库。编译警告保留为上游代码警告，不修改算法。
- 真实C++节点 + 合成ROS输入，独立domain177、loopback：收到1472条非空路径、735条速度消息、135条状态消息。属于**合成ROS组件测试**，不是实际导航到达成功。命令见`scripts/test_navigation_transport.py`。沙箱UDP受限产生警告，节点通过本机ROS transport完成消息交换；不宣称UDP外网验证通过。
- 静态本地import闭包无缺失，见`local_import_closure.json`。外部Isaac插件/系统依赖单独列出，不以AST检查代替运行验证。
- 53个私有资源文件已本地独立暂存，源asset不改；USD资产引用相对化，资源SHA完整；CLI本地preflight通过。私有资源默认不进入Git。

## C：未通过/未执行及原因

- **未执行完整Isaac单点闭环**：用户要求保持正在运行的训练，单实例锁保留，不开第二个仿真。移植后的Sim入口、云台yaw执行、XY禁止身体yaw后的到达性能尚需独占仿真验收。
- **未执行真实豆包API读取**：本轮环境没有ARK_API_KEY。安装真实SDK并完成严格解析/失败测试，不等同API端到端通过。
- **未验证全新GPU仿真环境安装、Orin或真机**；仿真环境实际版本仅作来源记录，不能冒充独立安装通过。
- usd-core24.11只读解析源资产：G1有3层、1个未解析MDL模块；场景1层47资产、4个未解析MDL模块。这些是Kit内置shader标识，不是凭空宣称文件丢失。必须在Kit验证着色器及渲染，独立usd-core检查没有标记全部通过。GroundPlane仍依赖Isaac平台官方资产。
- **未确认公开授权，未推送GitHub**。模型实际可用且本地LFS受管理，但公共下载资源/许可尚未完成；代码包不能被称为“公开后无需额外资源即可完整巡检”。

## 复现命令

按README安装后，在仓库外指定完整root：

```bash
g1-inspection --root "$RELEASE_ROOT" check-models
g1-inspection --root "$RELEASE_ROOT" replay --offline
python -m pytest "$RELEASE_ROOT/tests"
G1_CXX_RUNTIME=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 bash "$RELEASE_ROOT/scripts/build_navigation.sh"
```

完整任务需合法资源、已配置Isaac/ROS平台、有效API密钥且无占用仿真实例，再执行README的 `run` 与 `start_mission.py`。本报告没有伪造真实SUCCESS、碰撞率或新模型任务成功率。

## 本轮发布范围复核

保留完整 Demo 启动、Planner、导航、云台视角模型及 API 读数代码。排除所有仿真资产，两个策略/视觉权重保持原 SHA256，算法不变。38 项测试重新执行通过；模型离线回归另行复跑，结果见 upload_scope_check.json。未启动仿真或影响当前训练。用户不再要求以成绩作为上传门槛，真实闭环尚未验收的事实保留。

## GitHub 远程交付验证

完整代码及两个 LFS 权重已上传；从 GitHub 全新克隆，确认两个权重及 DINOv2 源码哈希匹配。从克隆构建并安装 wheel 到现有 CPU venv，从仓库外运行模型校验、38 项测试和4组禁止网络推理回归全部通过。本轮复用依赖环境，不宣称重新创建独立环境；未运行 Isaac 或真实 API。详见 github_delivery_validation.json。
