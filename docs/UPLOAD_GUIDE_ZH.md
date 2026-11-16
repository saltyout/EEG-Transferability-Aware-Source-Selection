# Windows 本地上传步骤

本次发布包含两个数据集的完整微调结果；Cho2017 旧阶段快照保留为历史记录。独立分支为 `codex/finetuning-release`，不直接覆盖 main。

1. 在本机打开 PowerShell，进入这份最终仓库目录。可在资源管理器中打开仓库文件夹后，在地址栏输入 powershell。
2. 确认分支及工作区：

```powershell
git switch codex/finetuning-release
git status
git log -1 --oneline
git remote -v
```

应看到当前分支正确、工作区干净，origin 指向 saltyout/EEG-Transferability-Aware-Source-Selection。文件已提交，不需要再 git add 或 git commit。

3. 上传分支：

```powershell
git push -u origin codex/finetuning-release
```

若弹出 GitHub 登录窗口，使用有仓库写权限的账号完成授权。不要把密码或访问令牌写进代码或聊天。若失败，保留报错信息，不要使用 force push。

4. 在浏览器打开目标 GitHub 仓库，通过 Pull Request 比较：base 为 main，compare 为 codex/finetuning-release。检查 README 中两个数据集都显示完整结果、Files changed 未包含 EEG/权重/缓存，再创建 PR。
5. 等待 CPU reproducibility 工作流完成。检查 PR 内容、测试结果及可能的冲突后，再选择合并。若有冲突，先解决并重跑检查，不要强行覆盖 main。
6. 合并后可在本地同步：

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
```

仓库公开目录刻意不包含服务器配置、原始 EEG、模型权重和逐试次预测。新通用 GPU 入口提供了运行实现，但未全量重跑验证；已完成的研究结果来自归档历史驱动。仓库尚未指定许可证，应由代码权利人决定，不能仅因公开可见就宣称允许任意复用。
