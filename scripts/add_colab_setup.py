import glob
import os
import nbformat


def get_colab_setup_code(ch: str) -> str:
    return f"""# === Google Colab 自動環境セットアップ ===
# ※ローカル環境では無視され、Colab環境でのみ自動でモジュールをインストールします
import sys, os

if 'google.colab' in sys.modules:
    REPO_NAME = "study-deep-learning"
    REPO_URL = f"https://github.com/miyayudai/{{REPO_NAME}}.git"
    TARGET_DIR = f"/content/{{REPO_NAME}}"
    CH_DIR = f"{{TARGET_DIR}}/{ch}"

    # 1. リポジトリのダウンロード (未ダウンロード時のみ)
    if not os.path.exists(TARGET_DIR):
        print("📦 リポジトリをダウンロード中...")
        res = os.system(f"git clone {{REPO_URL}} {{TARGET_DIR}} 2>&1")
        
        # Private リポジトリ等で認証が必要な場合の自動救済
        if res != 0 or not os.path.exists(TARGET_DIR):
            token = None
            try:
                from google.colab import userdata
                token = userdata.get('GITHUB_TOKEN')
            except Exception:
                pass
            
            if not token:
                import getpass
                print("🔒 リポジトリが非公開(Private)の場合、GitHub Personal Access Token (PAT) が必要です。")
                print("※公開(Public)リポジトリの場合はそのまま Enter を押してください。")
                token = getpass.getpass("GitHub Token (空欄可): ").strip()
            
            if token:
                auth_url = f"https://{{token}}@github.com/miyayudai/{{REPO_NAME}}.git"
                os.system(f"git clone {{auth_url}} {{TARGET_DIR}}")

    # 2. 依存パッケージとローカルモジュールのインストール
    print("⚙️ 環境セットアップ中...")
    if os.path.exists(TARGET_DIR):
        if TARGET_DIR not in sys.path:
            sys.path.insert(0, TARGET_DIR)
        req_path = os.path.join(TARGET_DIR, "requirements.txt")
        if os.path.exists(req_path):
            get_ipython().system(f"pip install -q -r {{req_path}}")
            get_ipython().system(f"pip install -q -e {{TARGET_DIR}}")
        if os.path.exists(CH_DIR):
            get_ipython().run_line_magic("cd", CH_DIR)
        else:
            get_ipython().run_line_magic("cd", TARGET_DIR)
        print("✅ 準備完了！このまま下のセルを実行できます。")
    else:
        print("⚠️ リポジトリのクローンがスキップされたため、必須パッケージを個別インストールします...")
        get_ipython().system("pip install -q pandas scikit-learn seaborn pillow")
        print("✅ パッケージインストール完了。")"""


def update_all_notebooks():
    def get_sort_key(p):
        d = p.split(os.sep)[0]
        return int(d) if d.isdigit() else (999 if d == "appendix" else 1000), p

    notebooks = sorted(glob.glob("*/*.ipynb"), key=get_sort_key)
    updated = 0

    for nb_path in notebooks:
        ch = nb_path.split(os.sep)[0]
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        setup_code = get_colab_setup_code(ch)

        # 既存のセル0がColabセットアップセルかどうか確認
        if nb.cells and "Google Colab 自動環境セットアップ" in nb.cells[0].source:
            # 既存のセル0を最新の堅牢なコードに更新
            nb.cells[0].source = setup_code
            updated += 1
        else:
            # セル0に新規挿入
            new_cell = nbformat.v4.new_code_cell(setup_code)
            new_cell.execution_count = None
            new_cell.outputs = []
            nb.cells.insert(0, new_cell)
            updated += 1

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    print(f"Colab setup updated: {updated} notebooks processed.")


if __name__ == "__main__":
    update_all_notebooks()
