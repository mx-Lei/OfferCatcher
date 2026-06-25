"""
Offer 捕手 - 学生求职匹配智能体
Alpha v0.4 — 匹配 + 优化建议
"""

import streamlit as st
import json
import os
import re

# ── 配置 ──────────────────────────────────────────────────
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3"


def get_api_key() -> str:
    """优先从 Streamlit Secrets / 环境变量读取 API Key，无则返回空"""
    try:
        return st.secrets["API_KEY"]
    except Exception:
        return os.getenv("API_KEY", "")

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="Offer 捕手",
    page_icon="🎯",
    layout="wide",
)

# ── 简历解析函数 ──────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> str:
    """用 PyMuPDF 提取 PDF 文本"""
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def parse_docx(file_bytes: bytes) -> str:
    """用 python-docx 提取 Word 文本"""
    import docx
    from io import BytesIO
    doc = docx.Document(BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def parse_resume(file_upload) -> str | None:
    """检测文件类型并调用对应解析器，返回纯文本"""
    if file_upload is None:
        return None
    raw = file_upload.read()
    fname = file_upload.name.lower()
    if fname.endswith(".pdf"):
        return parse_pdf(raw)
    elif fname.endswith(".docx"):
        return parse_docx(raw)
    else:
        st.error(f"不支持的文件格式: {fname}")
        return None


# ── 岗位数据 ──────────────────────────────────────────────

GENERIC_JD = (
    "具备该岗位相关专业知识与技能；有相关项目经验或实习经历优先；"
    "良好的学习能力、沟通能力和团队协作精神；本科及以上学历。"
)


def load_jobs() -> list[dict]:
    """从 jobs.json 加载岗位列表"""
    path = os.path.join(os.path.dirname(__file__), "jobs.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_job_by_id(job_id: int) -> dict | None:
    """按 ID 查找单个岗位（仅真实岗位）"""
    for j in load_jobs():
        if j["id"] == job_id:
            return j
    return None


def build_virtual_job(title: str) -> dict:
    """为用户输入的自定义岗位生成虚拟岗位，ID 用负数避免冲突"""
    return {
        "id": -1,
        "title": title,
        "company": "自定义岗位",
        "city": "全国",
        "salary": "面议",
        "requirements": GENERIC_JD,
        "tags": [title],
    }


def find_real_job(keyword: str) -> dict | None:
    """在 jobs.json 中按关键词查找真实岗位，无匹配返回 None"""
    if not keyword:
        return None
    kw = keyword.strip().lower()
    for j in load_jobs():
        if kw in j["title"].lower():
            return j
    for j in load_jobs():
        for t in j.get("tags", []):
            if kw in t.lower():
                return j
    return None


def get_effective_jobs(target_job: str) -> list[dict]:
    """获取用于匹配的岗位列表：真实岗位 + 必要时附加虚拟岗位"""
    jobs = list(load_jobs())
    if target_job and not find_real_job(target_job):
        # 用户输入了库里没有的岗位 → 构造虚拟岗位插入最前
        jobs.insert(0, build_virtual_job(target_job))
    return jobs


# ── LLM 调用 ──────────────────────────────────────────────

def call_llm(prompt: str, api_key: str) -> str:
    """调用 SiliconFlow DeepSeek API，返回模型原始输出"""
    import requests
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 4096,
        },
        timeout=90,
    )
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]


def extract_json(raw: str) -> str:
    """从 LLM 输出中提取纯 JSON 文本"""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw)
    if m:
        return m.group(1)
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        return m.group(0)
    return raw


def match_resume_to_jobs(resume_text: str, target_job: str, api_key: str) -> list[dict]:
    """核心匹配：调用 LLM 对简历与所有岗位（含虚拟岗位）逐一评分"""
    jobs = get_effective_jobs(target_job)

    jobs_desc = "\n\n".join([
        f"【岗位 {j['id']}】\n"
        f"名称: {j['title']}\n"
        f"公司: {j['company']}\n"
        f"城市: {j.get('city', '全国')}\n"
        f"薪资: {j.get('salary', '面议')}\n"
        f"要求: {j['requirements']}\n"
        f"标签: {', '.join(j.get('tags', []))}"
        for j in jobs
    ])

    prompt = f"""你是一位资深求职匹配顾问。请分析以下简历与各岗位的匹配程度。

【学生简历】
{resume_text[:3000]}

【岗位列表】
{jobs_desc}

请返回一个 JSON 数组（不要 markdown 代码块），格式如下：
[
  {{
    "job_id": 数字,
    "job_title": "岗位名称",
    "match_score": 0-100的整数,
    "strengths": ["匹配优势1", "优势2", "优势3"],
    "gaps": ["不足1", "不足2", "不足3"],
    "advice": "1-2句针对性提升建议"
  }}
]

只返回 JSON 数组。"""

    raw = call_llm(prompt, api_key)
    json_str = extract_json(raw)
    return json.loads(json_str)


def generate_optimization(resume_text: str, job_source, api_key: str) -> list[dict]:
    """针对指定岗位生成简历优化建议。

    job_source 可以是:
      - int: jobs.json 中的岗位 ID
      - dict: 虚拟岗位或任意岗位字典（必须含 title, requirements 字段）
    """
    if isinstance(job_source, int):
        job = get_job_by_id(job_source)
        if job is None:
            raise ValueError(f"未找到岗位 #{job_source}")
    else:
        job = job_source  # 虚拟岗位 dict

    prompt = f"""你是一位资深简历优化顾问。请根据岗位 JD 对学生的简历提出具体修改建议。

【目标岗位】
名称: {job['title']}
公司: {job.get('company', '')}
要求: {job['requirements']}
标签: {', '.join(job.get('tags', []))}

【学生简历全文】
{resume_text[:4000]}

请返回一个 JSON 数组（不要 markdown 代码块），包含 3-5 条优化建议，格式如下：
[
  {{
    "issue": "简历中的具体问题（一句话）",
    "suggestion": "如何修改（具体可操作的步骤）",
    "example": "修改后的示例文本（可直接使用）"
  }}
]

建议需覆盖：关键词缺失、经历量化不足、技能描述不够具体、项目经验不匹配等方面。
只返回 JSON 数组。"""

    raw = call_llm(prompt, api_key)
    json_str = extract_json(raw)
    return json.loads(json_str)


# ── Session 初始化 ────────────────────────────────────────
if "parsed_text" not in st.session_state:
    st.session_state.parsed_text = None
if "parsed_word_count" not in st.session_state:
    st.session_state.parsed_word_count = 0
if "match_results" not in st.session_state:
    st.session_state.match_results = None
if "optimization_suggestions" not in st.session_state:
    st.session_state.optimization_suggestions = None
if "optimizing_for_job_id" not in st.session_state:
    st.session_state.optimizing_for_job_id = None
if "optimizing_job_title" not in st.session_state:
    st.session_state.optimizing_job_title = ""

# ── 标题 ──────────────────────────────────────────────────
st.title("🎯 Offer 捕手")
st.caption("学生求职匹配智能体 — 上传简历，找到最适合你的岗位")

# ── 左侧边栏 ──────────────────────────────────────────────
with st.sidebar:
    st.header("📤 输入信息")

    uploaded_file = st.file_uploader(
        "上传简历",
        type=["pdf", "docx"],
        help="支持 PDF 或 Word 格式",
    )

    if uploaded_file is not None:
        text = parse_resume(uploaded_file)
        if text:
            wc = len(text.replace(" ", "").replace("\n", ""))
            st.session_state.parsed_text = text
            st.session_state.parsed_word_count = wc
            st.success(f"✅ 已解析，共 {wc} 字")
        elif st.session_state.parsed_text is None:
            st.warning("⚠️ 未能提取到文本内容")
    else:
        if st.session_state.parsed_text is not None:
            st.session_state.parsed_text = None
            st.session_state.parsed_word_count = 0
        st.info("请上传简历文件")

    target_job = st.text_input(
        "目标岗位",
        placeholder="例如：前端开发工程师、数据分析师",
    )

    api_key = st.text_input(
        "API Key",
        type="password",
        value=get_api_key() or "YOUR_API_KEY",
        help="留空则使用环境变量或 Streamlit Secrets 中的 API_KEY",
    ) if not get_api_key() else None

    # 若 Secrets 已配置，静默使用
    _effective_key = get_api_key() or (api_key if api_key != "YOUR_API_KEY" else "")
    if get_api_key():
        st.success("🔑 已从 Secrets 读取 API Key")

    match_btn = st.button("开始匹配", type="primary", use_container_width=True)

# ── 主区域 ────────────────────────────────────────────────
st.subheader("📊 匹配结果")

if match_btn:
    if not uploaded_file:
        st.error("请先上传简历文件")
    elif not _effective_key or _effective_key == "YOUR_API_KEY":
        st.error("请在侧边栏填入有效的 API Key")
    else:
        # 重置优化建议
        st.session_state.optimization_suggestions = None
        st.session_state.optimizing_for_job_id = None
        st.session_state.optimizing_job_title = ""

        with st.status("AI 匹配中…", expanded=True) as status:
            st.write("正在调用 DeepSeek 模型分析简历…")
            st.write("正在对比岗位需求…")
            st.write("正在生成匹配报告…")
            try:
                results = match_resume_to_jobs(
                    st.session_state.parsed_text,
                    target_job,
                    _effective_key,
                )
                st.session_state.match_results = results
                status.update(label=f"匹配完成！共分析 {len(results)} 个岗位",
                              state="complete", expanded=False)
            except Exception as e:
                status.update(label="匹配失败", state="error")
                st.error(f"API 调用失败: {e}")
                st.session_state.match_results = None

# ── 匹配结果展示 ──
if st.session_state.match_results:
    results = sorted(st.session_state.match_results,
                     key=lambda r: r.get("match_score", 0), reverse=True)

    for r in results:
        score = r.get("match_score", 0)

        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("匹配分数", f"{score}", delta=None)
            with col2:
                st.markdown(f"**{r.get('job_title', '-')}**")
                st.caption(f"🆔 岗位 #{r.get('job_id', '-')}  ·  💡 {r.get('advice', '')}")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ 优势**")
                for s in r.get("strengths", []):
                    st.markdown(f"- {s}")
            with c2:
                st.markdown("**⚠️ 不足**")
                for g in r.get("gaps", []):
                    st.markdown(f"- {g}")

    # ── 优化建议 ──
    st.markdown("---")
    st.subheader("🛠️ 简历优化建议")

    best_match = results[0]
    best_id = best_match.get("job_id")

    # 确定优化目标岗位：优先匹配 target_job 输入值（含虚拟岗位）
    if target_job:
        real = find_real_job(target_job)
        if real:
            opt_target_job = real
        else:
            opt_target_job = build_virtual_job(target_job)
    else:
        # 未输入目标岗位 → 用最高分结果
        opt_target_job = get_job_by_id(best_id) or build_virtual_job(best_match.get("job_title", "未知岗位"))

    opt_target_id = opt_target_job.get("id")
    opt_target_title = opt_target_job.get("title", "")

    opt_btn = st.button(
        f"查看优化建议（针对 {opt_target_title}）",
        type="secondary",
    )

    if opt_btn:
        with st.status("生成优化建议中…", expanded=True) as status:
            try:
                suggestions = generate_optimization(
                    st.session_state.parsed_text,
                    opt_target_job,
                    _effective_key,
                )
                st.session_state.optimization_suggestions = suggestions
                st.session_state.optimizing_for_job_id = opt_target_id
                st.session_state.optimizing_job_title = opt_target_title
                status.update(label=f"已生成 {len(suggestions)} 条建议",
                              state="complete", expanded=False)
            except Exception as e:
                status.update(label="生成失败", state="error")
                st.error(f"API 调用失败: {e}")

    # ── 展示优化建议 ──
    if st.session_state.optimization_suggestions:
        suggestions = st.session_state.optimization_suggestions
        title = st.session_state.optimizing_job_title or "目标岗位"
        st.caption(f"📌 针对 **{title}** 的优化建议")

        for idx, sug in enumerate(suggestions, 1):
            with st.expander(f"建议 {idx}：{sug.get('issue', '')}", expanded=(idx == 1)):
                st.markdown("**🔧 修改建议**")
                st.info(sug.get("suggestion", ""))
                st.markdown("**📝 参考示例**")
                st.code(sug.get("example", ""), language="text")

    if st.session_state.parsed_text:
        with st.expander("📄 简历解析预览", expanded=False):
            st.text(st.session_state.parsed_text[:2000])

else:
    st.info("请在左侧上传简历并填写目标岗位，然后点击「开始匹配」")
    st.markdown("""
    **使用说明：**
    1. 左侧上传简历（PDF / Word）
    2. 填写你的目标岗位
    3. 填入 SiliconFlow API Key
    4. 点击「开始匹配」
    5. 查看匹配结果 → 点击「查看优化建议」获取简历修改方案
    """)
