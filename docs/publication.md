# 发布范围与上传方式

用户已明确授权向 `https://github.com/chensamjyut/G1.git` 上传四部分：已选封装 GRPO 模型及运行代码、Planner、导航、读数；不再以评测成绩作为上传门槛。

## 上传内容

- 固定 `models/grpo_viewpoint.pt` 与所需 DINOv2 权重，均使用 Git LFS；不重新选择、训练或转换权重。
- 保留实际运行和接口代码、配置、依赖、测试及第三方许可证。`sim_runtime/` 是接入代码，不含场景资产。
- 不上传 `resources/private/`、USD/机器人/云台资产、工厂贴图、步行 ONNX、真实目标照片、训练输出、密钥或本机配置。仅保留四张合成测试 PNG。
- 原模型来源和历史评测保留作可追溯记录，不宣称云台 yaw 改接后的成绩已测得。
- 上传授权不是为全部自有代码另选开源许可证；保留已有版权声明，不新增项目许可证。

## 运行局限

资源由使用方合法另行提供。完整仿真闭环、云台接口后的闭环效果、真实 API 和真机能力仍未验收；这些不是本轮上传的成绩门槛。不得将离线测试通过描述为巡检任务完成。

## 上传步骤

只修改本独立发布仓库 remote；先检查远端历史，不覆盖默认分支或强推：

```bash
cd /path/to/github_release
git lfs fsck
git ls-remote https://github.com/chensamjyut/G1.git
# origin 尚不存在时添加
git remote add origin https://github.com/chensamjyut/G1.git
git fetch origin
git push -u origin HEAD:release/g1-inspection-minimal
```

## 远程交付校验

```bash
git clone --branch release/g1-inspection-minimal https://github.com/chensamjyut/G1.git G1-verify
cd G1-verify
git lfs pull
git lfs fsck
# 按 README 安装独立环境后
.venv/bin/g1-inspection --root "$PWD" check-models
.venv/bin/g1-inspection --root "$PWD" replay --offline
```

本机 Git/LFS 验证不等于 GitHub 上传或下载验证。实际状态见 `release_status.json`。

第三方来源说明：localPlanner 的原 package.xml 声明 BSD，但当前来源目录未提供独立完整许可文本；此事实仍保留，不能把上传授权表述为已补全上游许可。

## 本轮实际上传结果

远端 ls-remote 已成功，未返回任何 refs。实际 git push 在执行前被自动审批拒绝：需要用户明确批准本次向外部仓库发布源码与模型权重。未上传代码或 LFS 对象；没有执行 GitHub 全新克隆验证。待明确授权后上传 `release/g1-inspection-minimal`，不强推。
