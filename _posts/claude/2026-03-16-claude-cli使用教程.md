# Claude CLI 使用教程

## 📖 简介

Claude CLI（Claude Code）是 Anthropic 推出的命令行工具，让你可以在终端中直接与 Claude AI 协作完成编程任务。它支持文件读写、代码执行、Git 操作等功能，是一个强大的 AI 编程助手。

---

## 🚀 快速开始

### 1. 安装

```bash
# 使用 npm 安装
npm install -g @anthropic-ai/claude-code

# 或使用 yarn
yarn global add @anthropic-ai/claude-code
```

### 2. 认证

```bash
# 登录 Claude
claude login

# 这将打开浏览器让你完成认证
```

### 3. 开始使用

```bash
# 进入项目目录
cd your-project

# 启动 Claude CLI
claude
```

---

## 💡 基础用法

### 对话模式

启动 `claude` 后进入交互式对话模式：

```bash
$ claude
> 帮我创建一个 React 组件
```

### 单次查询模式

```bash
# 执行单个命令后退出
claude "帮我解释这个文件的功能"

# 带上下文执行
claude "重构这个函数" --file src/utils.js
```

### 项目上下文

Claude 会自动读取项目中的文件来理解上下文：

```bash
# 自动读取相关文件
claude "修复这个 bug"

# 指定文件
claude "分析这个文件" --file ./src/app.js

# 指定多个文件
claude "比较这两个文件" --file a.js --file b.js
```

---

## 📁 文件操作

### 读取文件

```bash
# 让 Claude 读取并解释文件
claude "解释这个文件的作用" --file README.md
```

### 写入/修改文件

```bash
# 让 Claude 创建新文件
claude "创建一个 index.html 文件，包含基本结构"

# 让 Claude 修改现有文件
claude "把这个函数改成异步的" --file src/api.js
```

### 文件模式

```bash
# 只读模式（Claude 不能修改文件）
claude --read-only

# 允许写入
claude --dangerously-skip-permissions
```

---

## 🔧 常用场景

### 1. 代码审查

```bash
claude "审查这个 PR 的代码变更" --file src/changes.js
```

### 2. Debug 调试

```bash
claude "帮我找出这个 bug 的原因" --file src/error.js
```

### 3. 重构代码

```bash
claude "重构这个函数，提高可读性" --file src/legacy.js
```

### 4. 生成测试

```bash
claude "为这个模块生成单元测试" --file src/module.js
```

### 5. 文档编写

```bash
claude "为这个 API 生成文档注释" --file src/api.js
```

### 6. Git 操作

```bash
claude "帮我写一个提交信息"
claude "查看最近的 git 变更"
```

---

## ⚙️ 高级功能

### 多文件上下文

```bash
# 同时提供多个文件作为上下文
claude "整合这些文件的功能" \
  --file src/a.js \
  --file src/b.js \
  --file src/c.js
```

### 管道输入

```bash
# 从管道读取输入
echo "console.log('hello')" | claude "格式化这段代码"
```

### 自定义系统提示

```bash
# 设置自定义系统提示
claude --system-prompt "你是一个资深的前端工程师"
```

---

## 🚀 高级用法（资深玩家必备）

### 1. MCP（Model Context Protocol）

MCP（Model Context Protocol）是 Anthropic 推出的开放协议，允许 Claude 与外部数据源和工具进行标准化集成。这是 Claude CLI **官方支持**的扩展方式。

#### ⚠️ 重要说明

> **Skills vs MCP 区别**：
> - **Skills** 是 OpenClaw 平台的技能系统（通过 clawhub 安装），**不是 Claude CLI 原生功能**
> - **MCP** 是 Claude 官方支持的协议，用于连接外部数据源和工具
> - 本教程中，Skills 相关功能仅在 OpenClaw 环境中可用；MCP 在所有 Claude 环境中通用

#### 配置 MCP Server

MCP 配置通过 `claude_desktop_config.json` 或环境变量进行设置。

**macOS/Linux 配置文件位置：**
```bash
~/.config/claude/claude_desktop_config.json
# 或
~/.claude/claude_desktop_config.json
```

**Windows 配置文件位置：**
```
%APPDATA%\Claude\claude_desktop_config.json
```

#### 配置文件示例

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/yourname/allowed-files"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "your_github_token"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/db"
      }
    }
  }
}
```

#### 安装 MCP Servers

```bash
# 文件系统服务器
npm install -g @modelcontextprotocol/server-filesystem

# GitHub 服务器
npm install -g @modelcontextprotocol/server-github

# PostgreSQL 服务器
npm install -g @modelcontextprotocol/server-postgres

# SQLite 服务器
npm install -g @modelcontextprotocol/server-sqlite

# 记忆服务器
npm install -g @modelcontextprotocol/server-memory

# Fetch 服务器（网页内容）
npm install -g @modelcontextprotocol/server-fetch
```

#### 官方 MCP Servers 列表

| Server | 用途 | NPM 包名 |
|--------|------|----------|
| 文件系统 | 访问指定目录的文件 | `@modelcontextprotocol/server-filesystem` |
| GitHub | GitHub API 集成 | `@modelcontextprotocol/server-github` |
| PostgreSQL | PostgreSQL 数据库 | `@modelcontextprotocol/server-postgres` |
| SQLite | SQLite 数据库 | `@modelcontextprotocol/server-sqlite` |
| 记忆 | 长期记忆存储 | `@modelcontextprotocol/server-memory` |
| Fetch | 网页内容获取 | `@modelcontextprotocol/server-fetch` |
| Slack | Slack 消息集成 | `@modelcontextprotocol/server-slack` |
| Google Drive | Google Drive 文件 | `@modelcontextprotocol/server-google-drive` |

#### 使用 MCP

```bash
# 启动 Claude（自动加载 MCP 配置）
claude

# 在对话中查看可用的 MCP 工具
/tools

# 使用 MCP 工具示例
# - "读取 ~/Documents 目录的文件列表"（使用 filesystem MCP）
# - "查询数据库中的用户表"（使用 postgres MCP）
# - "获取我的 GitHub 仓库列表"（使用 github MCP）
```

#### 安全配置

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem"],
      "config": {
        "allowedDirectories": ["/Users/yourname/projects", "/Users/yourname/docs"],
        "blockedPatterns": ["**/.env", "**/*.key", "**/secrets/**"]
      }
    }
  }
}
```

---

### 2. Shell 命令执行

Claude CLI 可以在获得授权后执行 Shell 命令来完成各种任务。

#### 启用 Shell 访问

```bash
# 启动时允许 Shell 命令（需要确认）
claude --dangerously-skip-permissions

# 或在配置文件中设置
echo '{"allowShell": true}' > .clauderc
```

#### 常用 Shell 命令场景

```bash
# 项目初始化
claude "初始化一个 Node.js 项目并安装必要依赖"

# Git 操作
claude "查看最近的提交历史并生成变更摘要"

# 文件操作
claude "在项目根目录创建一个 docs 文件夹"

# 构建和部署
claude "运行构建命令并检查是否有错误"
```

#### 安全限制

可以在配置中限制可执行的命令：

```json
{
  "shell": {
    "allowedCommands": ["npm", "yarn", "git", "ls", "cat", "mkdir", "cp", "mv"],
    "blockedCommands": ["rm -rf", "sudo", "curl | bash", "wget | bash"],
    "requireConfirm": ["npm publish", "git push", "docker rm"]
  }
}
```

---

### 3. 项目级配置文件（.clauderc）

在项目根目录创建 `.clauderc` 文件来定制 Claude 的行为：

```json
{
  "read_only": false,
  "system_prompt": "你是一个专业的 Node.js 开发者",
  "allowShell": true,
  "ignore": ["**/node_modules/**", "**/*.test.js"]
}
```

#### 自动化工作流

```bash
#!/bin/bash
# claude-ci.sh - CI/CD 集成脚本

# 代码审查
claude "审查这个 PR 的变更" --file $(git diff --name-only HEAD~1) > review.txt

# 生成变更日志
claude "根据 git 日志生成 changelog" --git-log >> CHANGELOG.md

# 更新文档
claude "根据代码变更更新 API 文档" --file src/api.js

echo "自动化完成！"
```

#### 定时任务集成

```bash
# 添加到 crontab
# 每天上午 9 点生成日报
0 9 * * * cd /path/to/project && claude "生成昨日开发日报" --git-log >> daily-report.md
```

---

### 4. 提示词技巧（Prompt Engineering）

更好的提示词能获得更好的结果。

#### 结构化提示词

```bash
# 好的提示词示例
claude "
任务：重构这个函数
要求：
1. 保持原有功能不变
2. 提高代码可读性
3. 添加 TypeScript 类型注解
4. 添加 JSDoc 注释
文件：src/utils.js
"
```

#### 上下文管理

```bash
# 提供清晰的上下文
claude "
背景：这是一个 React 项目，使用 TypeScript 和 Tailwind CSS
任务：创建一个新的按钮组件
要求：支持 variant（primary/secondary）、size（sm/md/lg）、disabled 状态
输出：组件文件 + Storybook 故事
"
```

#### 迭代式开发

```bash
# 第一步：生成初稿
claude "创建一个登录表单组件"

# 第二步：改进
claude "添加表单验证功能"

# 第三步：优化
claude "添加加载状态和错误处理"
```

---

### 5. 自动化与 CI/CD 集成

#### Shell 脚本自动化

```bash
#!/bin/bash
# claude-review.sh - 代码审查脚本

echo "开始代码审查..."

# 获取变更文件
CHANGED_FILES=$(git diff --name-only HEAD~1)

# 让 Claude 审查
claude "
请审查以下文件的变更：
$CHANGED_FILES

审查要点：
1. 代码质量问题
2. 潜在 bug
3. 性能问题
4. 安全漏洞
" --file $CHANGED_FILES > review-report.md

echo "审查完成！报告已保存到 review-report.md"
```

#### Git Hooks 集成

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "运行 Claude 代码检查..."

claude "检查暂存区的代码是否有明显问题" --file $(git diff --cached --name-only)

if [ $? -ne 0 ]; then
  echo "❌ 代码检查未通过"
  exit 1
fi

echo "✅ 代码检查通过"
```

#### GitHub Actions 集成

```yaml
# .github/workflows/claude-review.yml
name: Claude Code Review

on: [pull_request]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Claude CLI
        run: npm install -g @anthropic-ai/claude-code
      
      - name: Claude Review
        run: |
          claude "审查 PR 变更并输出报告" \
            --file $(git diff --name-only origin/main..HEAD) \
            > review.md
      
      - name: Upload Review
        uses: actions/upload-artifact@v3
        with:
          name: claude-review
          path: review.md
```

---

### 6. 最佳实践总结

#### 安全实践

| 实践 | 说明 |
|------|------|
| 使用只读模式 | 首次处理陌生项目时用 `--read-only` |
| 审查变更 | 确认 Claude 的文件修改建议 |
| 限制目录 | 在配置中限制可访问的目录 |
| 敏感文件 | 将 `.env`、密钥文件加入忽略列表 |
| 命令白名单 | 限制可执行的 Shell 命令 |

#### 效率实践

| 实践 | 说明 |
|------|------|
| 明确任务 | 提示词越具体，结果越准确 |
| 分步执行 | 复杂任务拆分成多个小步骤 |
| 利用上下文 | 提供相关文件和背景信息 |
| 保存会话 | 重要会话导出保存供后续参考 |
| 配置复用 | 项目配置 `.clauderc` 纳入版本控制 |

#### 完整配置示例（.clauderc）

```json
{
  "model": "claude-3.5-sonnet",
  "maxTokens": 8192,
  "readOnly": false,
  "allowShell": true,
  "ignore": [
    "**/node_modules/**",
    "**/dist/**",
    "**/*.test.js",
    "**/.env*",
    "**/*.log"
  ],
  "shell": {
    "allowedCommands": ["npm", "yarn", "git", "ls", "cat", "mkdir", "cp", "mv"],
    "blockedCommands": ["rm -rf /", "sudo", "curl | bash"]
  },
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

---

### 7. 常见问题排查

#### Claude 无法访问某些文件？

```bash
# 检查文件权限
ls -la path/to/file

# 检查配置中的 ignore 规则
cat .clauderc

# 尝试显式指定文件
claude "分析这个文件" --file path/to/file
```

#### MCP Server 无法启动？

```bash
# 检查 Node.js 版本
node --version

# 全局安装 MCP server
npm install -g @modelcontextprotocol/server-filesystem

# 测试 MCP server
npx @modelcontextprotocol/server-filesystem /tmp

# 检查配置文件语法
cat ~/.config/claude/claude_desktop_config.json | jq .
```

#### 响应速度慢？

```bash
# 减少上下文文件数量
claude "分析这个文件" --file src/main.js

# 限制 token 数量
claude --max-tokens 2048
```

#### 如何查看 Claude 执行了什么命令？

```bash
# 启用详细日志
claude --verbose

# 查看会话历史
claude --history

# 导出会话记录
claude --export-session <session-id> > session-log.json
```

---

## 🛡️ 安全与权限

### 权限模式

| 模式 | 说明 |
|------|------|
| `--read-only` | 只读模式，Claude 不能修改任何文件 |
| `--dangerously-skip-permissions` | 跳过所有权限确认（慎用） |
| 默认模式 | 修改文件前会请求确认 |

### 最佳实践

1. **首次使用建议用只读模式**，了解 Claude 的能力
2. **重要文件修改前备份**
3. **审查 Claude 的修改建议**后再确认
4. **不要在敏感目录使用** `--dangerously-skip-permissions`

---

## 📝 实用技巧

### 1. 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+C` | 中断当前操作 |
| `Ctrl+D` | 退出对话 |
| `Ctrl+L` | 清屏 |

### 3. 会话历史

```bash
# 查看历史会话
claude --history

# 恢复之前的会话
claude --resume <session-id>
```

---

## ❓ 常见问题

### Q: Claude 无法读取某些文件？
A: 检查文件权限，确保 Claude 有读取权限。

### Q: 如何取消正在进行的操作？
A: 按 `Ctrl+C` 中断。

### Q: 如何退出 Claude CLI？
A: 输入 `/exit` 或按 `Ctrl+D`。

### Q: 可以离线使用吗？
A: 不可以，需要网络连接访问 Anthropic API。

---

## 🔗 相关资源

- [官方文档](https://docs.anthropic.com/claude-code/)
- [GitHub 仓库](https://github.com/anthropics/claude-code)
- [社区讨论](https://discord.gg/anthropic)

---

*最后更新：2026-03-16*
