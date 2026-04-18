# PDF Font Subset

The PDF export uses [Noto Sans SC](https://fonts.google.com/noto/specimen/Noto+Sans+SC) (Simplified Chinese), licensed under the [SIL Open Font License 1.1](https://openfontlicense.org/).

## Files

- `public/fonts/NotoSansSC-Regular-subset.otf` — regular weight, ~70KB
- `public/fonts/NotoSansSC-Bold-subset.otf` — bold weight, ~70KB

Both files contain only the glyphs used by the app UI (derived from `src/i18n.ts`) plus common genetics/biology terms and punctuation — currently ~410 characters.

## Regenerating

Whenever new Chinese strings are added to `src/i18n.ts`, regenerate the character set and re-subset the fonts.

### Prerequisites

```bash
pip install fonttools brotli
```

### Steps

1. Regenerate the character list:

```bash
python -c "
import re, pathlib
src = pathlib.Path('src/i18n.ts').read_text(encoding='utf-8')
chars = set()
for ch in src:
    cp = ord(ch)
    if 0x4E00 <= cp <= 0x9FFF: chars.add(ch)          # CJK Unified
    elif 0x3000 <= cp <= 0x303F: chars.add(ch)        # CJK punctuation
    elif 0xFF00 <= cp <= 0xFFEF: chars.add(ch)        # Fullwidth
for cp in range(0x20, 0x7F): chars.add(chr(cp))       # printable ASCII
extras = '←→↑↓…—·•°％×÷≤≥±≠℃μαβγΔ'
for ch in extras: chars.add(ch)
bio = 'ATCGNatcgnIDRefQryMutInsDelSubFrameshiftIdentityCoverageCDSSampleDataset样本数据集比对参考突变插入缺失替换移码覆盖率一致性质量得分分析报告导出筛选排序状态错误警告完成正在结果识别未知不确定正确错误通过失败生物信息测序碱基氨基酸序列位置类型影响'
for ch in bio: chars.add(ch)
pathlib.Path('docs/fonts/common-chars.txt').write_text(''.join(sorted(chars)), encoding='utf-8')
print(f'wrote {len(chars)} chars')
"
```

2. Download source fonts (if not cached):

```bash
mkdir -p temp_fonts
curl -L -o temp_fonts/NotoSansSC-Regular.otf \
  https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf
curl -L -o temp_fonts/NotoSansSC-Bold.otf \
  https://github.com/notofonts/noto-cjk/raw/main/Sans/SubsetOTF/SC/NotoSansSC-Bold.otf
```

3. Subset:

```bash
python -m fontTools.subset temp_fonts/NotoSansSC-Regular.otf \
  --text-file=docs/fonts/common-chars.txt \
  --output-file=public/fonts/NotoSansSC-Regular-subset.otf \
  --no-hinting --desubroutinize

python -m fontTools.subset temp_fonts/NotoSansSC-Bold.otf \
  --text-file=docs/fonts/common-chars.txt \
  --output-file=public/fonts/NotoSansSC-Bold-subset.otf \
  --no-hinting --desubroutinize
```

4. Clean up:

```bash
rm -rf temp_fonts
```

5. Commit `public/fonts/*.otf` and `docs/fonts/common-chars.txt` together.
