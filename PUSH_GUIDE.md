# PUSH_GUIDE — 发布操作手册（甲方手动执行）

> 本仓库（开源发布目录，下称"仓库根"）已是 push-ready 状态：git 首提交与
> tag `v0.1.0` 在位，副本门禁（pytest / 敏感扫描 / demo 冒烟）已跑通并留档
> 于 `_release_checks\`（不入 git）。**push 是甲方的手动动作**：按本手册
> 逐步执行即可。每一步都可以停在原地等确认，没有不可逆操作。

## ⚠️ 步骤 0 · 专利时序警示（先读这一行）

**若拟就本项目申请专利，必须先取得专利申请日、再公开本仓库** —— GitHub
公开本身构成"自行公开"的现有技术，中国专利法第 24 条宽限期**不覆盖**此情形。
建议顺序：先交专利申请 → 拿到申请日 → 再执行下述 push。若确定不申请专利，
从步骤 1 继续。

## 步骤 1 · push 前机器扫描（每次发布前重跑）

```bash
cd /d <仓库根>
python scripts\release_sensitivity_scan.py     # 敏感词 + 密钥/路径扫描
python -m pytest tests -q                      # 测试套件
run_demo.cmd                                   # 一键 demo 冒烟
```

三项全绿（敏感扫描 = 零未豁免命中）才继续。扫描器自带豁免表
（`scripts/release_sensitivity_scan.py` 内 `EXEMPTIONS`），任何新增命中件
要么删除、要么给出书面理由并登记进豁免表。

机器扫描清单：零密钥/零 token/零个人信息（用户名、机器名、内部路径）/
零内部商务材料 / 零禁止分发数据（beishan 系文件已被 .gitignore 策略层拦截，
`git status` 中不应出现）。

## 步骤 2 · 定稿占位符（push 前一次性完成）

| 文件 | 占位符 | 需要定 |
|---|---|---|
| `LICENSE` | 版权行 `[Copyright holder ...]` | 导师/学校署名；定稿后删除文件尾注 |
| `CITATION.cff` | authors / repository-code | 作者与 ORCID、GitHub 账号 |
| `SECURITY.md` | `[SECURITY CONTACT ...]` | 漏洞报告邮箱 |
| `README.md` | （无强制占位） | 可选：CI 徽章换成真实账号 |

## 步骤 3 · GitHub 建仓（先私有，后公开）

1. 登录 GitHub → **New repository** → 名称建议 `fractureflow`；
2. **Private** 先建（私有验收期），验收通过后在
   Settings → General → Danger Zone → *Change visibility* 转公开；
3. 建仓时**不要**勾选任何初始化（README/.gitignore/license 均不要），
   避免与本地历史冲突。

## 步骤 4 · push 命令（逐条）

```bash
cd /d <仓库根>
git remote add origin https://github.com/<你的账号>/fractureflow.git
git push -u origin main
git push origin v0.1.0
```

> 若账号启用了 2FA，密码处填 Personal Access Token（Settings → Developer
> settings → Tokens）。SSH 方式同理（`git@github.com:<账号>/fractureflow.git`）。
>
> 署名说明：首提交作者是构建占位身份（`FractureFlow Release Builder`）。
> push 前如需改为真实署名（未 push 阶段可安全执行）：
> `git config user.name "<真实姓名>" && git config user.email "<真实邮箱>"`
> 然后 `git commit --amend --reset-author --no-edit` 并重打 tag：
> `git tag -fa v0.1.0 -m "v0.1.0 first public release (push-ready)"`。

## 步骤 5 · GitHub 侧配置（转公开前完成）

- **分支保护**：Settings → Branches → Add rule → `main`：勾选
  *Require a pull request before merging*（单人项目可只勾
  **Do not allow force pushes** 与 **Do not allow deletions**）。
- **Actions**：仓库已带 `.github/workflows/ci.yml`（每次 push 自动
  pytest 出绿勾），无需额外配置；首次运行在 Actions 页确认绿勾。
- **Issues / PR 模板**：已在 `.github/ISSUE_TEMPLATE/` 与
  `PULL_REQUEST_TEMPLATE.md`，可直接使用。
- **SECURITY.md**：已就位，GitHub 会自动在 Security 页展示。

## 步骤 6 · 论文投稿期匿名评审（双盲替代方案）

双盲评审需隐藏作者与单位。推荐流程：

1. 用 **anonymous.4open.science**：上传一份去掉 `CITATION.cff`、
   `CHANGELOG.md` 中署名信息、`LICENSE` 版权行的匿名副本（脚本化：
   匿名打包前删除/替换上述文件即可，其余内容与本仓库一致）；
2. 正式接收后再公开 GitHub 仓库并引用正式链接；
3. 论文中数据可用性声明写法：
   "Code available at [anonymized link] (review) / github.com/... (accepted)"。

## 步骤 7 · Zenodo DOI（建议在转公开后做）

1. zenodo.org 用 GitHub 账号登录 → 授权 **fractureflow** 仓库；
2. GitHub 发一个 Release（选已 push 的 `v0.1.0`），Zenodo 自动归档并
   发放 DOI；
3. 拿到 DOI 后回填 `CITATION.cff` 的 `identifiers` 字段并重 push。

## 版本与备份纪律

- 语义化 tag（`v0.1.0` 起）；每次发新版本：改 `CHANGELOG.md` → 打 tag →
  push tag →（可选）Zenodo 归档。
- **主项目树（本仓库的源工作区）永远是唯一权威**，
  GitHub 仅为发布镜像；本仓库不是开发工作区。
- **每次发布前重跑步骤 1 的三项扫描**，并把输出追加到
  `_release_checks\`（该目录不入 git）。
- 若主项目树后续有实质更新需要出库：重新走 R146 的构建流程（按 R145
  白名单重拷 → 重跑副本门禁 → 新提交），不要在发布仓里反向开发。
