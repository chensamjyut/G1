# 模型与执行契约

选择 `models/grpo_viewpoint.pt`，完整 `VisualStopResidualPolicy` state_dict；SHA256 `6356ea40de6fd02b6988adf751aa6f7f788b4240137ee2aedd6b2e1f95312af8`。checkpoint metadata 指定 `Multi-turn trajectory GRPO`、`s1_3m_r0001`，初始 SFT SHA256 为 `cc0550846b94505012bb66ce784ac53a30783ee0829cf1300ad837259875a306`。577 SFT 对应 `round11_low_head_replay` 封装。逐轮指标和真实 rollout policy_version 见 `model_selection.json`。选择依据为已有留存 GRPO rollouts 的最高成功率及平均最终 DINO 余弦，不是 checkpoint 文件名或最新编号。未完成同起点独立比较，不能称为全局最优或改接口后最佳。

只加载这一份策略权重和 DINOv2 ViT-B/14；不需要另加载 SFT 基座、adapter、tokenizer、processor、LLM。网络包括冻结 movement head、单调 Stop head 和 visual_stop_residual，保留现有源码定义。eval、inference_mode、确定性 argmax；原训练采样温度1.15不用于本次推理。

DINOv2 源码 vendoring，原 commit unknown；树哈希和算法定义见 manifest，运行只加载本地文件，不用 torch.hub 下载。DINO 输出 768 CLS 加 4×4 池化 patch tokens = 13056 维。图像 PIL RGB → bilinear resize 280×210 → float32 CHW /255 → mean [.485,.456,.406]、std [.229,.224,.225] normalization。不裁剪。

SIFT 是独立分支（`vision/sift_geometry.py`），不是 DINO 张量：文件直接 OpenCV 灰度读取（数组按RGB转灰），INTER_AREA resize 640×480；SIFT nfeatures=1500、contrastThreshold=.02；双向BF kNN k=2、ratio<.8且互相匹配；≥8对才估计基础矩阵RANSAC(2px,.995)和Homography RANSAC(3px)，输出61维几何特征。所有参数和失败分支原样保留，不替换为 DINO 或假特征。运行回归检查实际内点计数一致。相似度是 DINO 编码向量 cosine，clamp [-1,1]。

| id | 保留的模型 token | 发布执行含义 |
|---|---|---|
|0|MoveXPos|身体相对任务坐标 X+，0.3m平移步长，身体 yaw 指令0|
|1|MoveXNeg|任务 X−|
|2|MoveYPos|任务 Y+|
|3|MoveYNeg|任务 Y−|
|4|BodyYawLeft|**云台 pan +5°**，不调用导航/身体 yaw|
|5|BodyYawRight|**云台 pan −5°**|
|6|GimbalPitchUp|云台 tilt −5°，原约定正角低头|
|7|GimbalPitchDown|云台 tilt +5°|
|8|Stop|实际零速度指令和执行状态检查；随后等新RGB|

平移坐标变换、步长、云台限制和伺服容差见 `execution/configs/`；世界位移是 step×[sin(yaw)dx+cos(yaw)dy, −cos(yaw)dx+sin(yaw)dy]。不可用名称自行推测相机/身体坐标系。历史 yaw_step 参数为历史身体动作遗留，在部署 yaw 路由中不使用。

历史为最近8个实际已请求执行的 token，时间从旧到新，右侧 padding9；新 mission 为[]，encode为8个9。当前源码在执行结果写入后追加该 token，再检查失败；失败即结束，不再用失败历史推理。拒绝 START、相机缺帧不增加历史，安全停车不伪造Stop。读数间隔不插入假动作。策略原始输出与执行结果分别保存。

同一黑图对照会得到原始 Stop、DINO余弦1.0、SIFT内点0；这是相同 DINO输入造成，不是匹配失败后的默认成功值。不同无纹理图在本次实际回归为 GimbalPitchUp、相似度约0.641424、内点0。SIFT失效只说明该分支没有图像对应证据，不能单独推出DINO无效或任务成功。离线CLI始终 `task_success=false`；闭环只有实际 API 的结构化可读结果才能完成读数任务，Stop本身不认定任务完成。相机适配器检查图像shape/dtype及真实递增frame ID/time，不能用target替代current。没有把任何新增成功阈值或真机保持判断加入此包。
