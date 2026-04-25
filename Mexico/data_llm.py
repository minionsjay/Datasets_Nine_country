import asyncio
import json
import os
import csv
import pandas as pd
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt
from tqdm import tqdm

# ==========================================
# 1. 核心配置区
# ==========================================
deep_seek_API_BASE_URL = "https://api.deepseek.com" 
deep_seek_API_KEY = "sk-44ef59c0fe4e47e992fb4e4d985efe3f"
deep_seek_model = "deepseek-v4-flash"

Qianwen_api_key="sk-35805b6e284846f691c01e1b4caf4759"
Qianwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
Qianwen_model="qwen-plus-2025-07-28"

Xiaomi_api_key="sk-c4s1udb9y702t14eb05ijwfcurjfstaqlywlioxoo1rpp6ua"
Xiaomi_base_url="https://api.xiaomimimo.com/v1"
Xiaomi_model="mimo-v2-flash"

# 当前使用的模型 (可根据需要切换)
MODEL_NAME = deep_seek_model
API_BASE = deep_seek_API_BASE_URL
API_KEY = deep_seek_API_KEY

INPUT_CSV = "all_datasets_with_labels_dedup.csv"          # 原始数据
OUTPUT_CSV = f"./llm/{MODEL_NAME}/new_annotated_data.csv"            # 标注后的打分数据表
OUTPUT_LEXICON_CSV = f"./llm/{MODEL_NAME}/extracted_lexicon.csv"     # 提取的敏感词记录表

CONCURRENCY_LIMIT = 50      # 最大并发请求数
BATCH_SAVE_SIZE = 20        # 每处理 20 条强制存盘一次

# 印度尼西亚本地化维度
EXPECTED_KEYS = [
    "hate_speech", "false_info", "violence",
    "harassment", "obscenity", "illegal", "national_security"
]

client = AsyncOpenAI(api_key=API_KEY, base_url=API_BASE)
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# ==========================================
# 2. 合并任务的 System Prompt（印尼专属 + 解释 + 敏感词中文含义）
# ==========================================
SYSTEM_PROMPT = """你是一个专业的墨西哥多语言内容安全审核引擎。
你需要同时完成【违规打分】、【敏感词提取】以及【判定理由】三个任务。

【任务 1：违规打分（墨西哥法律框架）】
严格按照以下 7 个维度进行二元打分（命中为 1.0，未命中为 0.0）：
- hate_speech: 基于种族、宗教、性别、性取向等差异的仇恨言论或歧视。
  违反《联邦防止和消除歧视法》(LFPED)，禁止任何基于种族、性别或性取向的歧视与仇恨言论。
- false_info: 虚假信息或可能引发公众恐慌、操纵民主进程的内容。
  平时较少直接定罪，但大选期间受国家选举研究所(INE)严格监管，以防范选民操纵。
- violence: 煽动暴力、身体伤害、美化贩毒集团文化（Narcocultura）或展示极端帮派暴力。
  重点打击美化贩毒集团、展示斩首等极端暴力的视听内容。
- harassment: 针对个人的网络欺凌、人肉搜索、名誉诽谤、数字暴力或性别暴力。
  这是墨西哥的核心红线，依据《奥林匹亚法》(Ley Olimpia)，严厉打击网络性骚扰及数字暴力。
- obscenity: 淫秽、色情、露骨性内容或严重违背当地道德规范的行为，包括未经同意分享私密图像（复仇色情）。
  受《奥林匹亚法》管辖，复仇色情定义为联邦重罪。
- illegal: 在线赌博、欺诈、非法贷款、违禁品交易等网络犯罪。
  依据联邦刑法打击网络欺诈，限制未经内政部(SEGOB)许可的在线博彩宣传。
- national_security: 危害国家安全、煽动颠覆国家政权、恐怖主义袭击或破坏关键国家基础设施。
  依据墨西哥刑法典，打击煽动颠覆、恐怖主义及破坏关键基础设施的言论。

【任务 2：敏感词提取】
如果文本在任意维度违规，请提取出导致违规的核心“敏感词组（可以是两个或两个以上不同词组成的词组，用于过滤违规内容，词组内词语用&连接）/黑话”，并将其归入以下【8个一级分类】之一：
涉敏、涉政、未成年、价值观、宗教、低俗色情、暴力恐怖、违法。
同时，必须给出该敏感词组对应的中文意思（用中文简要解释该词组的含义）。
(如果没有敏感词，提取列表为空 []。)

【任务 3：判定理由】
对于每个命中（打分为 1.0）的维度，必须在 explanations 对象中用 1-2 句话说明为什么该文本属于这一类型，
引用具体的词语、表述或上下文，并指出违反的墨西哥法规或准则（例如 LFPED、Ley Olimpia、联邦刑法等）。
未命中的维度不需要出现在 explanations 中。

【输出格式要求】
必须且只能返回合法的 JSON 对象。请严格仿照以下格式：
{
  "hate_speech": 1.0,
  "false_info": 0.0,
  "violence": 0.0,
  "harassment": 0.0,
  "obscenity": 0.0,
  "illegal": 0.0,
  "national_security": 0.0,
  "explanations": {
    "hate_speech": "文本基于性取向进行贬低和攻击，构成歧视性仇恨言论，违反墨西哥LFPED法。"
  },
  "extracted_items": [
    {
      "word": "歧视性侮辱词&黑话",
      "category": "低俗色情",
      "meaning": "对特定群体的侮辱性称呼"
    }
  ]
}
如果没有任何维度命中（全0），explanations 必须为空对象 {}，extracted_items 必须为空列表 []。"""
# ==========================================
# 3. 异步请求与 JSON 解析
# ==========================================
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(5))
async def annotate_single_text(text: str, index: int) -> dict:
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"输入文本：{text}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            raw_result = response.choices[0].message.content
            parsed_json = json.loads(raw_result)
            
            # 校验打分字段完整性
            if not all(key in parsed_json for key in EXPECTED_KEYS):
                raise ValueError(f"JSON 缺少核心打分字段: {parsed_json}")
                 
            # 确保提取列表存在
            if "extracted_items" not in parsed_json:
                parsed_json["extracted_items"] = []
            # 确保 explanations 存在
            if "explanations" not in parsed_json:
                parsed_json["explanations"] = {}
                 
            return {"index": index, "labels": parsed_json, "status": "success"}
            
        except Exception as e:
            raise RuntimeError(f"Index {index} 处理失败: {str(e)}")

# ==========================================
# 4. 辅助函数：微批次安全落盘
# ==========================================
def save_batch_to_disk(results, df):
    new_rows = []
    lexicon_rows = []
    
    for res in results:
        if isinstance(res, Exception):
            continue
            
        idx = res["index"]
        labels = res["labels"]
        
        # 1. 组装主表数据 (打分 + 原因)
        row_data = {
            "index": idx,
            "text": df.at[idx, 'text'],
            "status": res["status"]
        }
        for key in EXPECTED_KEYS:
            row_data[key] = float(labels.get(key, 0.0))
        # 新增：将 explanations 保存为 JSON 字符串
        row_data["explanations"] = json.dumps(labels.get("explanations", {}), ensure_ascii=False)
        new_rows.append(row_data)
        
        # 2. 组装词库表数据 (敏感词 + 中文含义)
        for item in labels.get("extracted_items", []):
            word = str(item.get("word", "")).strip()
            category = str(item.get("category", "")).strip()
            meaning = str(item.get("meaning", "")).strip()
            if word and category:
                lexicon_rows.append({
                    "source_index": idx,
                    "word": word.lower(),
                    "category": category,
                    "meaning": meaning
                })

    # 追加写入主打分表
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        write_header = not os.path.exists(OUTPUT_CSV)
        new_df.to_csv(OUTPUT_CSV, mode='a', index=False, header=write_header, quoting=csv.QUOTE_MINIMAL)

    # 追加写入敏感词库表
    if lexicon_rows:
        lexicon_df = pd.DataFrame(lexicon_rows)
        write_lexicon_header = not os.path.exists(OUTPUT_LEXICON_CSV)
        lexicon_df.to_csv(OUTPUT_LEXICON_CSV, mode='a', index=False, header=write_lexicon_header, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)

# ==========================================
# 5. 主控流与微批次调度引擎
# ==========================================
async def main():
    print(f"🚀 启动印尼多任务自动流水线 (基于 {MODEL_NAME})...")
    
    # --- 1. 数据加载 ---
    if not os.path.exists(INPUT_CSV):
        print("未找到输入文件，正在生成测试用例...")
        df = pd.DataFrame({
            "text": [
                "I love this beautiful world!", 
                "Maldita puta, te vamos a matar.", 
                "Dasar anjing, orang Cina pulang sana!"
            ]
        })
        df.to_csv(INPUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        df = pd.read_csv(INPUT_CSV, on_bad_lines='warn', engine='python')

    # --- 2. 严格断点续传检测 ---
    processed_indices = set()
    if os.path.exists(OUTPUT_CSV):
        out_df = pd.read_csv(OUTPUT_CSV)
        if 'index' in out_df.columns:
            processed_indices = set(out_df['index'].tolist())
        print(f"📦 发现断点记录，跳过已处理的 {len(processed_indices)} 条数据...")

    unprocessed_df = df[~df.index.isin(processed_indices)]
    
    if unprocessed_df.empty:
        print("✅ 所有数据已处理完毕！")
        return

    print(f"⚙️ 剩余 {len(unprocessed_df)} 条任务，开始并发处理 (并发: {CONCURRENCY_LIMIT}, 每 {BATCH_SAVE_SIZE} 条存盘一次)...")
    
    # --- 3. 微批次切分与执行 ---
    all_indices = unprocessed_df.index.tolist()
    pbar = tqdm(total=len(all_indices), desc="总进度")
    
    for i in range(0, len(all_indices), BATCH_SAVE_SIZE):
        batch_indices = all_indices[i : i + BATCH_SAVE_SIZE]
        
        batch_tasks = []
        for idx in batch_indices:
            text = df.at[idx, 'text']
            batch_tasks.append(annotate_single_text(text, idx))
            
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # 立刻保存到硬盘
        save_batch_to_disk(batch_results, df)
        
        for res in batch_results:
            if isinstance(res, Exception):
                print(f"\n❌ [异常拦截] {res}")
                
        pbar.update(len(batch_indices))

    pbar.close()
    print(f"\n🎉 运行结束！")
    print(f"✔️ 打分结果已增量保存至: {OUTPUT_CSV}")
    print(f"✔️ 敏感词日志已增量保存至: {OUTPUT_LEXICON_CSV}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())