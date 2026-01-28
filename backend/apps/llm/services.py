"""
LLM Service for code review using various LLM providers
"""
import os
import logging
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with LLM providers
    """

    def __init__(self, request_id=None):
        self.request_id = request_id
        self._load_config()
        self.prompt_template = getattr(settings, 'GPT_MESSAGE',
                                     "请仔细审查以下代码变更，重点关注：\n1. 代码质量和最佳实践\n2. 潜在的bug和安全问题\n3. 性能优化建议\n4. 代码风格和可读性\n\n请提供具体的改进建议。")

    def _load_config(self):
        """
        从数据库加载 LLM 配置，如缺失则抛出 ImproperlyConfigured
        """
        try:
            from .models import LLMConfig
        except ImportError as exc:
            raise ImproperlyConfigured("无法导入 LLMConfig 模型，请确认 apps.llm 已正确安装") from exc

        llm_config = LLMConfig.objects.filter(is_active=True).first()

        if not llm_config:
            raise ImproperlyConfigured("未检测到有效的 LLM 配置，请在后台管理或接口中创建并启用一条 LLM 配置")

        self.provider = llm_config.provider
        self.api_key = llm_config.api_key
        self.api_base = llm_config.api_base
        self.model = llm_config.model
        self.config_source = "database"
        self.is_mock = self.provider == 'mock'

        api_key_status = "已配置" if self.api_key else "未配置"
        logger.info(
            f"[{self.request_id}] LLM配置加载成功 - 提供商:{self.provider}, 模型:{self.model}, API密钥:{api_key_status}"
        )

        if not self.is_mock:
            if not self.api_key:
                logger.warning(f"[{self.request_id}] LLM 配置缺少 API Key，后续请求可能失败")
            self._set_environment_variables()

    def _set_environment_variables(self):
        """
        将数据库配置设置为环境变量，供 LLM 库使用
        """
        if self.is_mock:
            return

        try:
            # 根据不同的提供商设置相应的环境变量
            if self.provider.lower() == 'openai':
                if self.api_key:
                    os.environ['OPENAI_API_KEY'] = self.api_key
                    logger.info(f"[{self.request_id}] 设置 OPENAI_API_KEY 环境变量")
                if self.api_base:
                    os.environ['OPENAI_API_BASE'] = self.api_base
                    logger.info(f"[{self.request_id}] 设置 OPENAI_API_BASE 环境变量")

            elif self.provider.lower() == 'deepseek':
                if self.api_key:
                    os.environ['DEEPSEEK_API_KEY'] = self.api_key
                    logger.info(f"[{self.request_id}] 设置 DEEPSEEK_API_KEY 环境变量")
                if self.api_base:
                    os.environ['DEEPSEEK_API_BASE'] = self.api_base
                    logger.info(f"[{self.request_id}] 设置 DEEPSEEK_API_BASE 环境变量")

            elif self.provider.lower() == 'claude':
                if self.api_key:
                    os.environ['ANTHROPIC_API_KEY'] = self.api_key
                    logger.info(f"[{self.request_id}] 设置 ANTHROPIC_API_KEY 环境变量")

            elif self.provider.lower() == 'gemini':
                if self.api_key:
                    os.environ['GOOGLE_API_KEY'] = self.api_key
                    logger.info(f"[{self.request_id}] 设置 GOOGLE_API_KEY 环境变量")

            elif self.provider.lower() == 'ollama':
                # Ollama 本地模型不需要 API Key
                if self.api_base:
                    os.environ['OLLAMA_API_BASE'] = self.api_base
                    logger.info(f"[{self.request_id}] 设置 OLLAMA_API_BASE 环境变量: {self.api_base}")

            # 通用配置，适用于所有提供商
            if self.model:
                os.environ['LLM_MODEL'] = self.model
                logger.info(f"[{self.request_id}] 设置 LLM_MODEL 环境变量: {self.model}")

        except Exception as e:
            logger.error(f"[{self.request_id}] 设置环境变量失败: {e}", exc_info=True)

    def review_code(self, code_context, mr_info=None, repo_path=None, commit_range=None, custom_prompt=None):
        """
        Review code using the configured LLM provider

        Args:
            code_context: 代码上下文（已弃用，保留用于向后兼容）
            mr_info: MR 信息字典
            repo_path: 本地仓库路径（使用 CLI 时必需）
            commit_range: Git 提交范围
            custom_prompt: 外部传入的自定义 prompt（优先级最高，来自项目配置）

        Returns:
            解析后的审查结果字典或错误消息字符串
        """
        import time
        start_time = time.time()

        logger.info(f"[{self.request_id}] 开始代码审查 - 使用提供商: {self.provider}")
        logger.info(f"[{self.request_id}] 仓库路径: {repo_path}")
        logger.info(f"[{self.request_id}] 提交范围: {commit_range}")

        # 检查是否提供了仓库路径（对于需要本地仓库的提供商）
        if not repo_path and self.provider in ['claude', 'ollama']:
            error_msg = f"代码审查失败：未提供仓库路径，{self.provider} 需要本地仓库"
            logger.error(f"[{self.request_id}] {error_msg}")
            return error_msg

        try:
            # 根据提供商选择相应的服务
            if self.provider.lower() == 'claude':
                return self._review_with_claude_cli(mr_info, repo_path, commit_range, custom_prompt, start_time)
            elif self.provider.lower() == 'ollama':
                return self._review_with_ollama(mr_info, repo_path, commit_range, custom_prompt, start_time)
            else:
                # 对于其他提供商（如 deepseek, openai 等），使用基于代码上下文的审查
                return self._review_with_generic_llm(mr_info, repo_path, commit_range, custom_prompt, start_time)

        except ImportError as e:
            logger.error(f"[{self.request_id}] 导入模块失败: {e}")
            return f"代码审查失败：缺少必要的模块\n错误详情: {str(e)}"
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[{self.request_id}] 代码审查异常 - 耗时:{elapsed_time:.2f}秒, 错误:{e}", exc_info=True)
            return f"代码审查失败: {str(e)}"

    def _review_with_claude_cli(self, mr_info, repo_path, commit_range, custom_prompt, start_time):
        """
        使用 Claude CLI 执行代码审查

        Args:
            mr_info: MR 信息字典
            repo_path: 本地仓库路径
            commit_range: Git 提交范围
            custom_prompt: 自定义 prompt
            start_time: 开始时间

        Returns:
            解析后的审查结果字典或错误消息字符串
        """
        from apps.review.claude_cli_service import ClaudeCliService
        from apps.review.review_result_parser import ReviewResultParser

        # 初始化 Claude CLI 服务
        cli_service = ClaudeCliService(request_id=self.request_id)

        # 验证 Claude CLI 安装
        is_valid, validation_error = cli_service.validate_cli_installation()
        if not is_valid:
            error_msg = f"代码审查失败：{validation_error}"
            logger.error(f"[{self.request_id}] {error_msg}")
            return error_msg

        # 构建最终使用的 prompt
        final_prompt = None
        if custom_prompt:
            logger.info(f"[{self.request_id}] 使用外部传入的自定义 Prompt (长度: {len(custom_prompt)})")
            final_prompt = custom_prompt
        elif mr_info:
            logger.info(f"[{self.request_id}] 使用系统默认 Prompt 构建逻辑")
            final_prompt = self._build_claude_cli_prompt(mr_info)
        else:
            logger.info(f"[{self.request_id}] 未提供任何 Prompt，使用 Claude CLI 默认行为")

        # 执行代码审查
        success, result_data, error = cli_service.review_code(
            repo_path=repo_path,
            custom_prompt=final_prompt,
            commit_range=commit_range
        )

        elapsed_time = time.time() - start_time

        if not success:
            error_msg = f"代码审查失败: {error}"
            logger.error(f"[{self.request_id}] {error_msg}")
            return error_msg

        # 解析审查结果
        parser = ReviewResultParser(request_id=self.request_id)
        parsed_result = parser.parse(result_data)

        logger.info(f"[{self.request_id}] 代码审查完成 - 耗时:{elapsed_time:.2f}秒")
        logger.info(f"[{self.request_id}] 评分:{parsed_result['score']}, 问题数:{len(parsed_result['issues'])}")

        return parsed_result

    def _review_with_ollama(self, mr_info, repo_path, commit_range, custom_prompt, start_time):
        """
        使用 Ollama 执行代码审查

        Args:
            mr_info: MR 信息字典
            repo_path: 本地仓库路径
            commit_range: Git 提交范围
            custom_prompt: 自定义 prompt
            start_time: 开始时间

        Returns:
            解析后的审查结果字典或错误消息字符串
        """
        from apps.review.ollama_service import OllamaService
        from apps.review.review_result_parser import ReviewResultParser

        # 初始化 Ollama 服务
        ollama_service = OllamaService(request_id=self.request_id)

        # 验证 Ollama 连接
        is_valid, validation_error = ollama_service.validate_ollama_connection()
        if not is_valid:
            error_msg = f"代码审查失败：{validation_error}"
            logger.error(f"[{self.request_id}] {error_msg}")
            return error_msg

        # 构建最终使用的 prompt
        final_prompt = None
        if custom_prompt:
            logger.info(f"[{self.request_id}] 使用外部传入的自定义 Prompt (长度: {len(custom_prompt)})")
            final_prompt = custom_prompt
        elif mr_info:
            logger.info(f"[{self.request_id}] 使用系统默认 Prompt 构建逻辑")
            final_prompt = self._build_claude_cli_prompt(mr_info)  # 复用相同的 prompt 结构
        else:
            logger.info(f"[{self.request_id}] 未提供任何 Prompt，使用 Ollama 默认行为")

        # 执行代码审查
        success, result_data, error = ollama_service.review_code(
            repo_path=repo_path,
            custom_prompt=final_prompt,
            commit_range=commit_range
        )

        elapsed_time = time.time() - start_time

        if not success:
            error_msg = f"代码审查失败: {error}"
            logger.error(f"[{self.request_id}] {error_msg}")
            return error_msg

        # 解析审查结果
        parser = ReviewResultParser(request_id=self.request_id)
        parsed_result = parser.parse(result_data)

        logger.info(f"[{self.request_id}] 代码审查完成 - 耗时:{elapsed_time:.2f}秒")
        logger.info(f"[{self.request_id}] 评分:{parsed_result['score']}, 问题数:{len(parsed_result['issues'])}")

        return parsed_result

    def _build_claude_cli_prompt(self, mr_info):
        """
        构建 Claude CLI 的审查提示

        Args:
            mr_info: MR 信息字典

        Returns:
            格式化的提示文本
        """
        prompt_parts = [
            f"请对这个 Merge Request 进行详细的代码审查：",
            f"",
            f"**MR 信息**",
            f"- 标题: {mr_info.get('title', '未知')}",
            f"- 作者: {mr_info.get('author', '未知')}",
            f"- 源分支: {mr_info.get('source_branch', '未知')}",
            f"- 目标分支: {mr_info.get('target_branch', '未知')}",
        ]

        if mr_info.get('description'):
            prompt_parts.append(f"- 描述: {mr_info['description']}")

        prompt_parts.extend([
            f"",
            f"**审查要求**",
            f"请对本次代码变更进行全面、深入的分析，提供一份详细、有条理、结构清晰的审查报告。",
            f"报告需要具有专业性和实用性，不仅要指出问题，还要提供具体的改进方案。",
            f"",
            f"## 📋 审查维度",
            f"",
            f"### 1. **安全风险评估** 🔒",
            f"- **认证与授权**：权限校验、身份验证、会话管理",
            f"- **输入验证**：参数校验、类型检查、边界条件处理",
            f"- **数据安全**：SQL 注入、XSS、CSRF、路径遍历等漏洞",
            f"- **敏感信息**：硬编码密钥、密码泄露、调试信息暴露",
            f"- **API 安全**：接口权限、数据传输安全、错误信息泄露",
            f"",
            f"### 2. **代码质量评估** 💎",
            f"- **架构设计**：模块划分、依赖关系、设计模式使用",
            f"- **代码规范**：命名约定、代码格式、注释质量",
            f"- **可读性**：逻辑清晰度、变量命名、函数设计",
            f"- **一致性**：代码风格统一、接口设计一致",
            f"- **复杂度**：圈复杂度、嵌套层级、函数长度",
            f"",
            f"### 3. **性能分析与优化** ⚡",
            f"- **算法效率**：时间复杂度、空间复杂度优化",
            f"- **数据库性能**：查询优化、索引使用、N+1 问题",
            f"- **资源管理**：内存使用、文件操作、网络请求",
            f"- **并发处理**：线程安全、异步处理、锁机制",
            f"- **缓存策略**：缓存设计、失效机制、缓存命中率",
            f"",
            f"### 4. **可维护性与扩展性** 🔧",
            f"- **模块化程度**：单一职责、接口抽象、依赖注入",
            f"- **错误处理**：异常处理机制、错误恢复、日志记录",
            f"- **配置管理**：配置分离、环境管理、配置验证",
            f"- **测试友好**：单元测试、集成测试、Mock 设计",
            f"- **文档完整性**：API 文档、注释质量、架构文档",
            f"",
            f"### 5. **业务逻辑审查** 🎯",
            f"- **功能完整性**：需求覆盖、边界条件、异常场景",
            f"- **数据一致性**：事务处理、数据校验、状态管理",
            f"- **用户体验**：响应时间、错误提示、操作流程",
            f"- **业务规则**：逻辑正确性、规则完整性、约束检查",
            f"",
            f"## 📝 报告输出格式要求",
            f"",
            f"请严格按照以下结构输出报告：",
            f"",
            f"```markdown",
            f"# 📊 代码审查报告",
            f"",
            f"## 📋 审查概述",
            f"- **MR 标题**：[填写 MR 标题]",
            f"- **审查范围**：[列出主要变更文件]",
            f"- **变更类型**：[新功能/修复/重构/优化]",
            f"- **风险等级**：[高/中/低/无]",
            f"- **总体评分**：[0-100 分]",
            f"",
            f"## 🎯 关键发现",
            f"",
            f"### 🟢 优秀实践",
            f"- 列出本次变更中的亮点和良好实践",
            f"",
            f"### 🔴 关键问题",
            f"- 列出必须修复的严重问题",
            f"- 每个问题包含：问题描述、影响、修复建议",
            f"",
            f"## 🔍 详细分析",
            f"",
            f"### 🔒 安全问题分析",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **问题描述**：详细说明问题",
            f"  - **风险评估**：潜在影响和安全风险",
            f"  - **修复建议**：具体解决方案",
            f"  - **代码示例**：修复前后对比",
            f"",
            f"### 💎 代码质量问题",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **问题描述**：详细说明问题",
            f"  - **影响分析**：对可维护性、可读性的影响",
            f"  - **改进建议**：具体的重构建议",
            f"  - **代码示例**：改进方案",
            f"",
            f"### ⚡ 性能问题分析",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **问题描述**：性能瓶颈详细说明",
            f"  - **性能影响**：对系统性能的具体影响",
            f"  - **优化方案**：详细的优化策略",
            f"  - **预期收益**：性能提升预估",
            f"",
            f"### 🔧 架构与设计问题",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **设计问题**：架构层面的问题",
            f"  - **影响分析**：对系统架构的影响",
            f"  - **重构建议**：设计改进方案",
            f"",
            f"### 🎯 业务逻辑问题",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **逻辑问题**：业务逻辑的缺陷",
            f"  - **业务影响**：对业务功能的影响",
            f"  - **修复方案**：逻辑修正建议",
            f"",
            f"## 📈 评分详情",
            f"",
            f"| 维度 | 评分 | 说明 |",
            f"|------|------|------|",
            f"| 安全性 | [0-25] | 安全问题严重程度 |",
            f"| 代码质量 | [0-25] | 代码规范和质量问题 |",
            f"| 性能表现 | [0-25] | 性能问题和优化空间 |",
            f"| 可维护性 | [0-25] | 架构设计和维护成本 |",
            f"| **总分** | **[0-100]** | **综合评价** |",
            f"",
            f"## 🚀 行动建议",
            f"",
            f"### 📋 必须修复（立即处理）",
            f"- [ ] [严重问题 1] - 修复优先级：P0",
            f"- [ ] [严重问题 2] - 修复优先级：P0",
            f"",
            f"### ⚠️ 建议修复（尽快处理）",
            f"- [ ] [重要问题 1] - 修复优先级：P1",
            f"- [ ] [重要问题 2] - 修复优先级：P1",
            f"",
            f"### 💡 优化建议（后续处理）",
            f"- [ ] [优化项 1] - 修复优先级：P2",
            f"- [ ] [优化项 2] - 修复优先级：P2",
            f"",
            f"## 📚 学习资源",
            f"- 针对本次发现的问题，提供相关的最佳实践和学习资料",
            f"",
            f"## ✅ 审查结论",
            f"",
            f"**综合评价**：[对本次变更的整体评价]",
            f"",
            f"**风险提示**：[主要风险点提醒]",
            f"",
            f"**合并建议**：[建议：立即合并/修复后合并/不建议合并]",
            f"",
            f"---",
            f"*审查完成时间：[当前时间]*",
            f"*审查工具：Claude AI*",
            f"```",
            f"",
            f"## ⚠️ 重要说明",
            f"",
            f"1. **必须包含具体文件路径和行号**，如：`backend/apps/models.py:45`",
            f"2. **每个问题都要提供具体的代码示例**，展示修复前后的对比",
            f"3. **严格按照模板格式输出**，确保报告的完整性和可读性",
            f"4. **评分要客观公正**，基于实际代码质量进行评估",
            f"5. **建议要具体可行**，避免模糊的描述",
            f"6. **风险等级标记**：🔴严重（必须修复）、🟠高危（建议尽快修复）、🟡中危（可稍后修复）、🟢低危（优化建议）"
        ])

        return '\n'.join(prompt_parts)

    def _review_with_openai(self, code_context, start_time):
        """
        Fallback to OpenAI client if UnionLLM is not available
        """
        import time
        try:
            from openai import OpenAI

            prompt = self._build_prompt(code_context)

            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None
            )

            logger.info(f"[{self.request_id}] 使用OpenAI客户端开始调用API")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位资深的代码审查专家,擅长发现代码中的问题并提供建设性的改进建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            elapsed_time = time.time() - start_time
            review_content = response.choices[0].message.content
            logger.info(f"[{self.request_id}] OpenAI客户端代码审查完成 - 耗时:{elapsed_time:.2f}秒, 响应长度:{len(review_content)}字符")
            return review_content

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[{self.request_id}] OpenAI客户端调用失败 - 耗时:{elapsed_time:.2f}秒, 错误:{e}", exc_info=True)
            return f"代码审查失败: {str(e)}"

    def _review_with_generic_llm(self, mr_info, repo_path, commit_range, custom_prompt, start_time):
        """
        使用通用 LLM 提供商进行代码审查（如 deepseek, openai 等）

        Args:
            mr_info: MR 信息字典
            repo_path: 本地仓库路径
            commit_range: Git 提交范围
            custom_prompt: 自定义 prompt
            start_time: 开始时间

        Returns:
            解析后的审查结果字典或错误消息字符串
        """
        import time
        try:
            # 从本地仓库获取代码变更
            code_context = self._get_code_context_from_repo(repo_path, commit_range)
            
            if not code_context:
                error_msg = "代码审查失败：未获取到代码变更内容"
                logger.error(f"[{self.request_id}] {error_msg}")
                return error_msg

            # 构建最终使用的 prompt
            final_prompt = None
            if custom_prompt:
                logger.info(f"[{self.request_id}] 使用外部传入的自定义 Prompt (长度: {len(custom_prompt)})")
                final_prompt = custom_prompt
            elif mr_info:
                logger.info(f"[{self.request_id}] 使用系统默认 Prompt 构建逻辑")
                final_prompt = self._build_generic_llm_prompt(mr_info, code_context)
            else:
                logger.info(f"[{self.request_id}] 未提供任何 Prompt，使用默认行为")
                final_prompt = self._build_prompt(code_context)

            # 调用 LLM API 进行审查
            review_content = self._call_llm_api(final_prompt)

            elapsed_time = time.time() - start_time

            # 解析审查结果
            from apps.review.review_result_parser import ReviewResultParser
            parser = ReviewResultParser(request_id=self.request_id)
            
            # 构造符合解析器期望的字典格式
            review_data = {
                'result': review_content,
                'duration_ms': int(elapsed_time * 1000)
            }
            parsed_result = parser.parse(review_data)

            logger.info(f"[{self.request_id}] 代码审查完成 - 耗时:{elapsed_time:.2f}秒")
            logger.info(f"[{self.request_id}] 评分:{parsed_result['score']}, 问题数:{len(parsed_result.get('issues', []))}")

            return parsed_result

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[{self.request_id}] 代码审查异常 - 耗时:{elapsed_time:.2f}秒, 错误:{e}", exc_info=True)
            return f"代码审查失败: {str(e)}"

    def _get_code_context_from_repo(self, repo_path, commit_range):
        """
        从本地仓库获取代码变更上下文

        Args:
            repo_path: 本地仓库路径
            commit_range: Git 提交范围

        Returns:
            代码变更上下文字符串
        """
        import subprocess
        import os

        try:
            # 检查仓库路径是否存在
            if not os.path.exists(repo_path):
                logger.error(f"[{self.request_id}] 仓库路径不存在: {repo_path}")
                return None
            
            # 检查 commit_range 是否为空
            if not commit_range:
                logger.warning(f"[{self.request_id}] commit_range 为空，尝试使用默认分支对比")
                # 尝试获取当前分支和默认分支
                try:
                    # 获取当前分支
                    current_branch = subprocess.run(
                        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    ).stdout.strip()
                    
                    # 获取默认分支 (通常是 main 或 master)
                    default_branch = subprocess.run(
                        ['git', 'remote', 'show', 'origin'],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=10
                    ).stdout
                    
                    # 从输出中提取默认分支
                    import re
                    match = re.search(r"HEAD branch: (\w+)", default_branch)
                    if match:
                        default_branch = match.group(1)
                        commit_range = f"{default_branch}..{current_branch}"
                        logger.info(f"[{self.request_id}] 使用默认分支对比: {commit_range}")
                    else:
                        logger.warning(f"[{self.request_id}] 无法获取默认分支，使用空范围")
                except Exception as e:
                    logger.warning(f"[{self.request_id}] 获取分支信息失败: {e}")
            
            # 备选方案列表
            # 先获取当前分支，用于后续的命令构建
            current_branch = ""
            try:
                current_branch = subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                ).stdout.strip()
            except Exception as e:
                logger.warning(f"[{self.request_id}] 获取当前分支失败: {e}")
            
            alternative_methods = [
                # 方法 1: 使用指定的 commit_range
                (f"git diff {commit_range}" if commit_range else "git diff", 
                 ['git', 'diff', commit_range] if commit_range else ['git', 'diff'], 
                 "指定范围"),
                
                # 方法 2: 获取最近 5 次提交
                ("git log -p -5", 
                 ['git', 'log', '-p', '-5'], 
                 "最近5次提交"),
                
                # 方法 3: 获取工作区和暂存区的所有变更
                ("git diff HEAD", 
                 ['git', 'diff', 'HEAD'], 
                 "工作区与HEAD对比"),
                
                # 方法 4: 获取所有未推送的变更
                ("git diff origin/$(git rev-parse --abbrev-ref HEAD)", 
                 ['git', 'diff', f"origin/{current_branch}"] if current_branch else ['git', 'diff'], 
                 "未推送变更"),
                
                # 方法 5: 获取最近一次提交与上一次提交的差异
                ("git diff HEAD~1 HEAD", 
                 ['git', 'diff', 'HEAD~1', 'HEAD'], 
                 "最近两次提交对比"),
            ]
            
            # 尝试所有备选方案
            for cmd_desc, cmd, method_name in alternative_methods:
                try:
                    logger.info(f"[{self.request_id}] 执行 {method_name}: {cmd_desc}")
                    
                    # 在仓库目录中执行命令
                    result = subprocess.run(
                        cmd,
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode == 0:
                        diff_content = result.stdout
                        
                        if diff_content:
                            logger.info(f"[{self.request_id}] 从{method_name}获取到代码变更，长度: {len(diff_content)} 字符")
                            # 限制内容长度
                            if len(diff_content) > 20000:
                                diff_content = diff_content[:20000] + "\n\n...(内容过长已截断)"
                            return diff_content
                        else:
                            logger.warning(f"[{self.request_id}] {method_name} 返回空结果")
                    else:
                        logger.warning(f"[{self.request_id}] {method_name} 执行失败: {result.stderr}")
                except Exception as e:
                    logger.warning(f"[{self.request_id}] {method_name} 执行异常: {e}")
            
            # 所有备选方案都失败，尝试获取当前分支信息并返回基本信息
            logger.error(f"[{self.request_id}] 所有获取代码变更的方法都失败")
            try:
                # 获取当前分支信息
                branch_info = subprocess.run(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                ).stdout.strip()
                
                # 获取最新提交信息
                commit_info = subprocess.run(
                    ['git', 'log', '-1', '--pretty=%B'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                ).stdout.strip()
                
                # 构建基本信息
                basic_info = f"# 仓库信息\n\n" \
                           f"- 当前分支: {branch_info}\n" \
                           f"- 最新提交: {commit_info}\n" \
                           f"- 注意: 无法获取代码变更详情，请检查仓库状态\n"
                
                logger.info(f"[{self.request_id}] 返回仓库基本信息")
                return basic_info
            except Exception as e:
                logger.error(f"[{self.request_id}] 获取仓库基本信息失败: {e}")
                return None

        except Exception as e:
            logger.error(f"[{self.request_id}] 获取代码上下文失败: {e}", exc_info=True)
            return None

    def _build_generic_llm_prompt(self, mr_info, code_context):
        """
        构建通用 LLM 的审查提示

        Args:
            mr_info: MR 信息字典
            code_context: 代码变更上下文

        Returns:
            格式化的提示文本
        """
        prompt_parts = [
            f"请对这个 Merge Request 进行详细的代码审查：",
            f"",
            f"**MR 信息**",
            f"- 标题: {mr_info.get('title', '未知')}",
            f"- 作者: {mr_info.get('author', '未知')}",
            f"- 源分支: {mr_info.get('source_branch', '未知')}",
            f"- 目标分支: {mr_info.get('target_branch', '未知')}",
        ]

        if mr_info.get('description'):
            prompt_parts.append(f"- 描述: {mr_info['description']}")

        prompt_parts.extend([
            f"",
            f"**审查要求**",
            f"请对本次代码变更进行全面、深入的分析，提供一份详细、有条理、结构清晰的审查报告。",
            f"报告需要具有专业性和实用性，不仅要指出问题，还要提供具体的改进方案。",
            f"",
            f"## 📋 审查维度",
            f"",
            f"### 1. **安全风险评估** 🔒",
            f"- **认证与授权**：权限校验、身份验证、会话管理",
            f"- **输入验证**：参数校验、类型检查、边界条件处理",
            f"- **数据安全**：SQL 注入、XSS、CSRF、路径遍历等漏洞",
            f"- **敏感信息**：硬编码密钥、密码泄露、调试信息暴露",
            f"- **API 安全**：接口权限、数据传输安全、错误信息泄露",
            f"",
            f"### 2. **代码质量评估** 💎",
            f"- **架构设计**：模块划分、依赖关系、设计模式使用",
            f"- **代码规范**：命名约定、代码格式、注释质量",
            f"- **可读性**：逻辑清晰度、变量命名、函数设计",
            f"- **一致性**：代码风格统一、接口设计一致",
            f"- **复杂度**：圈复杂度、嵌套层级、函数长度",
            f"",
            f"### 3. **性能分析与优化** ⚡",
            f"- **算法效率**：时间复杂度、空间复杂度优化",
            f"- **数据库性能**：查询优化、索引使用、N+1 问题",
            f"- **资源管理**：内存使用、文件操作、网络请求",
            f"- **并发处理**：线程安全、异步处理、锁机制",
            f"- **缓存策略**：缓存设计、失效机制、缓存命中率",
            f"",
            f"### 4. **可维护性与扩展性** 🔧",
            f"- **模块化程度**：单一职责、接口抽象、依赖注入",
            f"- **错误处理**：异常处理机制、错误恢复、日志记录",
            f"- **配置管理**：配置分离、环境管理、配置验证",
            f"- **测试友好**：单元测试、集成测试、Mock 设计",
            f"- **文档完整性**：API 文档、注释质量、架构文档",
            f"",
            f"### 5. **业务逻辑审查** 🎯",
            f"- **功能完整性**：需求覆盖、边界条件、异常场景",
            f"- **数据一致性**：事务处理、数据校验、状态管理",
            f"- **用户体验**：响应时间、错误提示、操作流程",
            f"- **业务规则**：逻辑正确性、规则完整性、约束检查",
            f"",
            f"## 📝 报告输出格式要求",
            f"",
            f"请严格按照以下结构输出报告：",
            f"",
            f"```markdown",
            f"# 📊 代码审查报告",
            f"",
            f"## 📋 审查概述",
            f"- **MR 标题**：[填写 MR 标题]",
            f"- **审查范围**：[列出主要变更文件]",
            f"- **变更类型**：[新功能/修复/重构/优化]",
            f"- **风险等级**：[高/中/低/无]",
            f"- **总体评分**：[0-100 分]",
            f"",
            f"## 🎯 关键发现",
            f"",
            f"### 🟢 优秀实践",
            f"- 列出本次变更中的亮点和良好实践",
            f"",
            f"### 🔴 关键问题",
            f"- 列出必须修复的严重问题",
            f"- 每个问题包含：问题描述、影响、修复建议",
            f"",
            f"## 🔍 详细分析",
            f"",
            f"### 🔒 安全问题分析",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **问题描述**：详细说明问题",
            f"  - **风险评估**：潜在影响和安全风险",
            f"  - **修复建议**：具体解决方案",
            f"  - **代码示例**：修复前后对比",
            f"",
            f"### 💎 代码质量问题",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **问题描述**：详细说明问题",
            f"  - **影响分析**：对可维护性、可读性的影响",
            f"  - **改进建议**：具体的重构建议",
            f"  - **代码示例**：改进方案",
            f"",
            f"### ⚡ 性能问题分析",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **问题描述**：性能瓶颈详细说明",
            f"  - **性能影响**：对系统性能的具体影响",
            f"  - **优化方案**：详细的优化策略",
            f"  - **预期收益**：性能提升预估",
            f"",
            f"### 🔧 架构与设计问题",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **设计问题**：架构层面的问题",
            f"  - **影响分析**：对系统架构的影响",
            f"  - **重构建议**：设计改进方案",
            f"",
            f"### 🎯 业务逻辑问题",
            f"- [问题 1] 🔴/🟠/🟡/🟢 标题",
            f"  - **文件**：`文件名:行号`",
            f"  - **逻辑问题**：业务逻辑的缺陷",
            f"  - **业务影响**：对业务功能的影响",
            f"  - **修复方案**：逻辑修正建议",
            f"",
            f"## 📈 评分详情",
            f"",
            f"| 维度 | 评分 | 说明 |",
            f"|------|------|------|",
            f"| 安全性 | [0-25] | 安全问题严重程度 |",
            f"| 代码质量 | [0-25] | 代码规范和质量问题 |",
            f"| 性能表现 | [0-25] | 性能问题和优化空间 |",
            f"| 可维护性 | [0-25] | 架构设计和维护成本 |",
            f"| **总分** | **[0-100]** | **综合评价** |",
            f"",
            f"## 🚀 行动建议",
            f"",
            f"### 📋 必须修复（立即处理）",
            f"- [ ] [严重问题 1] - 修复优先级：P0",
            f"- [ ] [严重问题 2] - 修复优先级：P0",
            f"",
            f"### ⚠️ 建议修复（尽快处理）",
            f"- [ ] [重要问题 1] - 修复优先级：P1",
            f"- [ ] [重要问题 2] - 修复优先级：P1",
            f"",
            f"### 💡 优化建议（后续处理）",
            f"- [ ] [优化项 1] - 修复优先级：P2",
            f"- [ ] [优化项 2] - 修复优先级：P2",
            f"",
            f"## 📚 学习资源",
            f"- 针对本次发现的问题，提供相关的最佳实践和学习资料",
            f"",
            f"## ✅ 审查结论",
            f"",
            f"**综合评价**：[对本次变更的整体评价]",
            f"",
            f"**风险提示**：[主要风险点提醒]",
            f"",
            f"**合并建议**：[建议：立即合并/修复后合并/不建议合并]",
            f"",
            f"---",
            f"*审查完成时间：[当前时间]*",
            f"*审查工具：{self.provider} AI*",
            f"```",
            f"",
            f"## ⚠️ 重要说明",
            f"",
            f"1. **必须包含具体文件路径和行号**，如：`backend/apps/models.py:45`",
            f"2. **每个问题都要提供具体的代码示例**，展示修复前后的对比",
            f"3. **严格按照模板格式输出**，确保报告的完整性和可读性",
            f"4. **评分要客观公正**，基于实际代码质量进行评估",
            f"5. **建议要具体可行**，避免模糊的描述",
            f"6. **风险等级标记**：🔴严重（必须修复）、🟠高危（建议尽快修复）、🟡中危（可稍后修复）、🟢低危（优化建议）",
            f"",
            f"## 代码变更内容",
            f"",
            f"```diff",
            f"{code_context}",
            f"```"
        ])

        return '\n'.join(prompt_parts)

    def _call_llm_api(self, prompt):
        """
        调用 LLM API 进行代码审查

        Args:
            prompt: 审查提示

        Returns:
            审查结果文本
        """
        try:
            # 直接使用提供商特定的 SDK
            logger.info(f"[{self.request_id}] 使用提供商特定的 SDK 调用 {self.provider} API")
            
            if self.provider.lower() == 'openai':
                return self._call_openai_api(prompt)
            elif self.provider.lower() == 'deepseek':
                return self._call_deepseek_api(prompt)
            elif self.provider.lower() == 'gemini':
                return self._call_gemini_api(prompt)
            else:
                # 对于其他提供商，返回错误
                error_msg = f"代码审查失败：提供商 {self.provider} 暂不支持 API 调用"
                logger.error(f"[{self.request_id}] {error_msg}")
                return error_msg

        except Exception as e:
            error_msg = f"代码审查失败：LLM API 调用失败 - {str(e)}"
            logger.error(f"[{self.request_id}] {error_msg}", exc_info=True)
            return error_msg

    def _call_openai_api(self, prompt):
        """
        调用 OpenAI API
        """
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base if self.api_base else None
            )
            logger.info(f"[{self.request_id}] 使用 OpenAI 客户端开始调用 API")
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位资深的代码审查专家,擅长发现代码中的问题并提供建设性的改进建议。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[{self.request_id}] OpenAI API 调用失败: {e}", exc_info=True)
            raise

    def _call_deepseek_api(self, prompt):
        """
        调用 DeepSeek API
        """
        try:
            import requests
            import json

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是一位资深的代码审查专家,擅长发现代码中的问题并提供建设性的改进建议。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 4000
            }

            api_base = self.api_base if self.api_base else "https://api.deepseek.com"
            url = f"{api_base}/chat/completions"

            logger.info(f"[{self.request_id}] 使用 DeepSeek API 开始调用")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"[{self.request_id}] DeepSeek API 调用失败: {e}", exc_info=True)
            raise

    def _call_gemini_api(self, prompt):
        """
        调用 Gemini API
        """
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            logger.info(f"[{self.request_id}] 使用 Gemini API 开始调用")
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 4000
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"[{self.request_id}] Gemini API 调用失败: {e}", exc_info=True)
            raise

    def _build_prompt(self, code_context):
        """
        Build the review prompt with code context
        """
        return f"{self.prompt_template}\n\n## 代码变更内容:\n\n{code_context}"

