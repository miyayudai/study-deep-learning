import glob
import os
import nbformat


def get_colab_setup_code(ch: str) -> str:
    return f"""# === Google Colab 自動環境セットアップ ===
# ※ローカル環境では無視され、Colab環境でのみ自動でモジュールをインストールします
import sys, os
if 'google.colab' in sys.modules:
    if not os.path.exists('/content/study-deep-learning'):
        print("リポジトリをダウンロード中...")
        !git clone https://github.com/miyayudai/study-deep-learning.git > /dev/null 2>&1
    
    print("必要なモジュールをインストール中...")
    %cd /content/study-deep-learning
    !pip install -q -r requirements.txt
    !pip install -q -e .
    
    print("作業ディレクトリをセットアップ中...")
    %cd /content/study-deep-learning/{ch}
    print("準備完了！このまま下のセルを実行できます。")"""


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

        if nb.cells and "Google Colab 自動環境セットアップ" in nb.cells[0].source:
            nb.cells[0].source = setup_code
            updated += 1
        else:
            new_cell = nbformat.v4.new_code_cell(setup_code)
            new_cell.execution_count = None
            new_cell.outputs = []
            nb.cells.insert(0, new_cell)
            updated += 1

        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

    print(f"Colab setup reverted to clean format: {updated} notebooks processed.")


if __name__ == "__main__":
    update_all_notebooks()
