from huggingface_hub import HfApi

# 配置
REPO_ID = "Chr1sN/BowlAndCup"  # 您的仓库 ID
TAG = "v3.0"                   # 必须与 info.json 中的 codebase_version 一致

print(f"正在为 {REPO_ID} 添加标签 {TAG} ...")

try:
    api = HfApi()
    api.create_tag(
        repo_id=REPO_ID,
        tag=TAG,
        repo_type="dataset",
        exist_ok=True  # 如果标签已存在也不会报错
    )
    print("✅ 标签添加成功！")
    print("现在您可以重新运行 compute_norm_stats 命令了。")

except Exception as e:
    print(f"❌ 操作失败: {e}")