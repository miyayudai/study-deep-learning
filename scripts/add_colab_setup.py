import glob, os, nbformat

def add_colab_setup_to_all_notebooks():
    notebooks = sorted(glob.glob('*/*.ipynb'), key=lambda p: (int(p.split('/')[0]) if p.split('/')[0].isdigit() else 999, p))
    updated = 0
    skipped = 0

    for nb_path in notebooks:
        ch = nb_path.split('/')[0]
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Check if already has colab setup in first 3 cells
        already_has = False
        for cell in nb.cells[:3]:
            src = cell.source
            if 'google.colab' in src and 'my_DeepLearning' in src:
                already_has = True
                break
                
        if already_has:
            skipped += 1
            continue
            
        setup_code = f"""# === Google Colab 自動環境セットアップ ===
# ※ローカル環境では無視され、Colab環境でのみ自動でモジュールをインストールします
import sys, os
if 'google.colab' in sys.modules:
    if not os.path.exists('/content/my_DeepLearning'):
        print("リポジトリをダウンロード中...")
        !git clone https://github.com/miyayudai/my_DeepLearning.git > /dev/null 2>&1
    
    print("必要なモジュールをインストール中...")
    %cd /content/my_DeepLearning
    !pip install -q -r requirements.txt
    !pip install -q -e .
    
    print("作業ディレクトリをセットアップ中...")
    %cd /content/my_DeepLearning/{ch}
    print("準備完了！このまま下のセルを実行できます。")"""
        
        new_cell = nbformat.v4.new_code_cell(setup_code)
        new_cell.execution_count = None
        new_cell.outputs = []
        
        nb.cells.insert(0, new_cell)
        
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        updated += 1

    print(f"Colab setup check: {updated} updated, {skipped} already configured.")

if __name__ == "__main__":
    add_colab_setup_to_all_notebooks()
