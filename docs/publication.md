# GitHub 发布门禁与命令

目标由用户指定为 `https://github.com/chensamjyut/G1.git`。本轮只准备本地独立仓库，未推送、未修改源仓库remote。不是对GitHub远程下载的验证。

## 当前阻塞

1. 自有planner/策略源码和权重没有可证明的统一公开许可，不擅自替作者选LICENSE。需要确认对目标仓库的上传授权、这些新增源码和权重的公开范围。
2. `resources/private/`包含工厂、目标截图及G1自定义USD/模型；未公开。工厂包受 NVIDIA Omniverse content terms约束，条款副本在 `third_party/licenses/factory_terms.txt`。确认合法分发或改为有权访问的私有资源交付，不能假造公开下载地址。
3. 原 localPlanner 声明BSD，但此工作目录没有独立完整许可证文本；发布前需要核实上游来源及完整许可文本。serial自带MIT许可已保留；Unitree派生代码保留Apache-2.0；DINOv2 Apache-2.0许可已保留（无CLIP依赖，不打包CLIP实现）。
4. 新执行接口下的真实Isaac闭环及API读数未验收，详见验证报告。不能把控制流单元测试或ROS合成输入写成真实巡检成功。

## 审核解除后执行

不要强推或覆盖现有远端历史。以下操作仅针对本独立目录：

```bash
cd /path/to/github_release
git status --short
git lfs fsck
git lfs ls-files
git remote -v
# 若origin不存在才添加：
git remote add origin https://github.com/chensamjyut/G1.git
git ls-remote origin
git fetch origin
# 检查远程内容/历史后，推到新分支供审查，不覆盖默认分支：
git push -u origin HEAD:release/g1-inspection-minimal
```

LFS pre-push hook会上传当前提交需要的两个权重；如失败则发布未完成。默认不推任何resources/private、config/*.local.json、.env或日志。检查 GitHub LFS配额和权限后，从远程重新clone、`git lfs pull`、模型校验、离线基准复核，才可称远端交付通过：

```bash
git clone --branch release/g1-inspection-minimal https://github.com/chensamjyut/G1.git G1-verify
cd G1-verify
git lfs pull
# 按README创建独立环境，再运行check-models / replay --offline。
```

本地提交身份使用自动化 `Release preparation <release@invalid>`，避免把工作站原有私人身份配置写入新仓库提交；这不是对原作者的许可替代或署名替换，所有源码版权声明仍保留。
