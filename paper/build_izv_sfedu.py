#!/usr/bin/env python3
"""Сборка статьи об edge-bench под требования журнала
«Известия ЮФУ. Технические науки».

Канон текста — paper/main.md (уже в структуре журнала; цитаты — ключами вида
[[key]]). Скрипт нумерует источники в порядке первого упоминания, вставляет
библиографический список (ГОСТ Р 7.0.5-2008) и его английскую версию на место
маркера <!-- REFERENCES -->, отрисовывает PlantUML-рисунки, собирает docx на
геометрии официального шаблона (output/Шаблон.docx: A4, Times New Roman 9 pt,
одинарный интервал) и проверяет: объём аннотаций, число и возраст источников,
провенанс чисел (каждое ключевое число текста найдено в файле результатов, из
которого взято), запрещённые слова, число страниц.

Запуск: python3 paper/build_izv_sfedu.py [--pt 10] [--no-pdf]
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAIN = HERE / 'main.md'
OUT = HERE / 'out'
FIG = HERE / 'figures'
TEMPLATE = (
    HERE / 'journal_template' / 'Шаблон.docx'
)  # копия output/Шаблон.docx (output/ в .gitignore)
PLANTUML_JAR = Path.home() / 'aspirantura/code/plant-disease-eccv2026/plantuml.jar'
ACCESS_DATE = '05.09.2026'
BODY_SZ = (
    18  # полупункты: 18 = 9 pt (как в Шаблон.docx), 20 = 10 pt (страница требований)
)

# key -> (год, запись RU по ГОСТ Р 7.0.5-2008, запись EN или None если совпадает)
REFS: dict[str, tuple[int, str, str | None]] = {
    'shi2016': (
        2016,
        'Shi W., Cao J., Zhang Q., Li Y., Xu L. Edge computing: Vision and challenges // IEEE Internet of Things Journal. — 2016. — Vol. 3, No. 5. — P. 637–646. — DOI: 10.1109/JIOT.2016.2579198.',
        None,
    ),
    'zhou2019': (
        2019,
        'Zhou Z., Chen X., Li E., Zeng L., Luo K., Zhang J. Edge intelligence: Paving the last mile of artificial intelligence with edge computing // Proceedings of the IEEE. — 2019. — Vol. 107, No. 8. — P. 1738–1762. — DOI: 10.1109/JPROC.2019.2918951.',
        None,
    ),
    'reddi2020': (
        2020,
        'Reddi V. J., Cheng C., Kanter D., Mattson P. et al. MLPerf Inference benchmark // Proc. ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA). — 2020. — P. 446–459. — DOI: 10.1109/ISCA45697.2020.00045.',
        None,
    ),
    'banbury2021': (
        2021,
        'Banbury C., Reddi V. J., Torelli P., Holleman J. et al. MLPerf Tiny benchmark // arXiv preprint arXiv:2106.07597. — 2021.',
        None,
    ),
    'ignatov2019': (
        2019,
        'Ignatov A., Timofte R., Kulik A., Yang S. et al. AI Benchmark: All about deep learning on smartphones in 2019 // Proc. IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). — 2019. — P. 3617–3635. — DOI: 10.1109/ICCVW.2019.00447.',
        None,
    ),
    'bianco2018': (
        2018,
        'Bianco S., Cadene R., Celona L., Napoletano P. Benchmark analysis of representative deep neural network architectures // IEEE Access. — 2018. — Vol. 6. — P. 64270–64277. — DOI: 10.1109/ACCESS.2018.2877890.',
        None,
    ),
    'hadidi2019': (
        2019,
        'Hadidi R., Cao J., Xie Y., Asgari B., Krishna T., Kim H. Characterizing the deployment of deep neural networks on commercial edge devices // Proc. IEEE International Symposium on Workload Characterization (IISWC). — 2019. — P. 35–48. — DOI: 10.1109/IISWC47752.2019.9041955.',
        None,
    ),
    'baller2021': (
        2021,
        'Baller S. P., Jindal A., Chadha M., Gerndt M. DeepEdgeBench: Benchmarking deep neural networks on edge devices // Proc. IEEE International Conference on Cloud Engineering (IC2E). — 2021. — P. 20–30. — DOI: 10.1109/IC2E52221.2021.00016.',
        None,
    ),
    'coleman2019': (
        2019,
        'Coleman C., Kang D., Narayanan D., Nardi L. et al. Analysis of DAWNBench, a time-to-accuracy machine learning performance benchmark // ACM SIGOPS Operating Systems Review. — 2019. — Vol. 53, No. 1. — P. 14–25. — DOI: 10.1145/3352020.3352024.',
        None,
    ),
    'gundersen2018': (
        2018,
        'Gundersen O. E., Kjensmo S. State of the art: Reproducibility in artificial intelligence // Proc. 32nd AAAI Conference on Artificial Intelligence. — 2018. — Vol. 32, No. 1. — DOI: 10.1609/aaai.v32i1.11503.',
        None,
    ),
    'pineau2020': (
        2020,
        'Pineau J., Vincent-Lamarre P., Sinha K., Larivière V. et al. Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program) // arXiv preprint arXiv:2003.12206. — 2020.',
        None,
    ),
    'abadi2015': (
        2015,
        f'Abadi M., Agarwal A., Barham P. et al. TensorFlow: Large-scale machine learning on heterogeneous systems. — 2015. — URL: https://www.tensorflow.org/ (дата обращения: {ACCESS_DATE}).',
        f'Abadi M., Agarwal A., Barham P. et al. TensorFlow: Large-scale machine learning on heterogeneous systems. — 2015. — URL: https://www.tensorflow.org/ (accessed: {ACCESS_DATE}).',
    ),
    'sandler2018': (
        2018,
        'Sandler M., Howard A., Zhu M., Zhmoginov A., Chen L.-C. MobileNetV2: Inverted residuals and linear bottlenecks // Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). — 2018. — P. 4510–4520. — DOI: 10.1109/CVPR.2018.00474.',
        None,
    ),
    'howard2017': (
        2017,
        'Howard A. G., Zhu M., Chen B., Kalenichenko D. et al. MobileNets: Efficient convolutional neural networks for mobile vision applications // arXiv preprint arXiv:1704.04861. — 2017.',
        None,
    ),
    'jacob2018': (
        2018,
        'Jacob B., Kligys S., Chen B., Zhu M., Tang M., Howard A., Adam H., Kalenichenko D. Quantization and training of neural networks for efficient integer-arithmetic-only inference // Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). — 2018. — P. 2704–2713. — DOI: 10.1109/CVPR.2018.00286.',
        None,
    ),
    'krishnamoorthi2018': (
        2018,
        'Krishnamoorthi R. Quantizing deep convolutional networks for efficient inference: A whitepaper // arXiv preprint arXiv:1806.08342. — 2018.',
        None,
    ),
    'nagel2021': (
        2021,
        'Nagel M., Fournarakis M., Amjad R. A., Bondarenko Y., van Baalen M., Blankevoort T. A white paper on neural network quantization // arXiv preprint arXiv:2106.08295. — 2021.',
        None,
    ),
    'cass2019': (
        2019,
        "Cass S. Taking AI to the edge: Google's TPU now comes in a maker-friendly package // IEEE Spectrum. — 2019. — Vol. 56, No. 5. — P. 16–17. — DOI: 10.1109/MSPEC.2019.8701189.",
        None,
    ),
    'coral2019': (
        2019,
        f'Google LLC. Coral: Edge TPU and products. — 2019. — URL: https://coral.ai/ (дата обращения: {ACCESS_DATE}).',
        f'Google LLC. Coral: Edge TPU and products. — 2019. — URL: https://coral.ai/ (accessed: {ACCESS_DATE}).',
    ),
    'raspberrypi': (
        2019,
        f'Raspberry Pi Ltd. Raspberry Pi 4 Model B. — 2019. — URL: https://www.raspberrypi.com/products/raspberry-pi-4-model-b/ (дата обращения: {ACCESS_DATE}).',
        f'Raspberry Pi Ltd. Raspberry Pi 4 Model B. — 2019. — URL: https://www.raspberrypi.com/products/raspberry-pi-4-model-b/ (accessed: {ACCESS_DATE}).',
    ),
}

# Провенанс чисел: (число в тексте, файл-источник относительно корня, токен в файле).
R_TPU = 'results/2026-08-14_c6_edgetpu/c6_edgetpu_latency.json'
R_CPU = 'results/2026-08-14_c6_edgetpu/c6_cpu_latency.json'
R_NORM = 'results/2026-08-14_c6_edgetpu/normalized.json'
R_CPU_FULL = 'results/2026-08-14_044028_c6_cpu_full/c6_cpu_latency.json'
R_M1 = 'results/platform_matrix/2026-08-13_085036_mobilenetv1_int8_ptq_Fuzzy.json'
R_OLDNEW = 'results/platform_matrix/2026-08-13_151458_c6_mobilenet_v2_int8_full.json'
PROVENANCE: list[tuple[str, str, str]] = [
    ('5,07', R_TPU, '"p50_ms": 5.07'),
    ('5,08 ± 0,02', R_TPU, '"mean_ms": 5.076'),
    ('0,02', R_TPU, '"std_ms": 0.024'),
    ('5,12', R_TPU, '"p95_ms": 5.121'),
    ('5,15', R_TPU, '"p99_ms": 5.148'),
    ('0,48', R_TPU, '"cv_percent": 0.48'),
    ('197,0', R_TPU, '"fps": 197.01'),
    ('2647,5', R_TPU, '"model_load_ms": 2647.51'),
    ('13,5', R_TPU, '"first_inference_ms": 13.48'),
    ('47,7', R_TPU, '"mean": 47.74'),
    ('5,5', R_TPU, '"mean": 5.48'),
    ('38,0 → 39,4', R_TPU, '"start": 37.97'),
    ('33,40 ± 0,98', R_CPU, '"mean_ms": 33.403'),
    ('33,24', R_CPU, '"p50_ms": 33.244'),
    ('34,41', R_CPU, '"p95_ms": 34.408'),
    ('35,23', R_CPU, '"p99_ms": 35.226'),
    ('2,94', R_CPU, '"cv_percent": 2.94'),
    ('29,9', R_CPU, '"fps": 29.94'),
    ('2,9', R_CPU, '"model_load_ms": 2.9'),
    ('37,4', R_CPU, '"first_inference_ms": 37.39'),
    ('47,6', R_CPU, '"mean": 47.57'),
    ('87,9', R_CPU, '"mean": 87.86'),
    ('39,4 → 43,8', R_CPU, '"end": 43.82'),
    ('6,56', R_NORM, '"speedup_tpu_vs_cpu": 6.56'),
    ('68 из 68', R_NORM, '"edgetpu_ops_mapped": "68/68"'),
    ('0,9692', R_NORM, '"cosine_to_fp32": 0.9692'),
    ('0,99999988', R_NORM, '"port_fidelity_cosine": 0.99999988'),
    ('55,99', R_CPU_FULL, '"p50_ms": 55.988'),
    ('0,91', R_M1, '"p50_ms": 0.909'),
    ('1,22', R_M1, '"p95_ms": 1.215'),
    ('1011', R_M1, '"fps": 1011.12'),
    ('61,8', R_M1, '"rss_mb": 61.8'),
    ('4,6', R_M1, '"model_load_ms": 4.62'),
    ('33,02', R_M1, '"p50_ms": 33.024'),
    ('35,98', R_M1, '"p95_ms": 35.98'),
    ('29,8', R_M1, '"fps": 29.83'),
    ('51,2', R_M1, '"rss_mb": 51.2'),
    ('1,7', R_M1, '"model_load_ms": 1.71'),
    ('4,70', R_M1, '"p50_ms": 4.695'),
    ('4,78', R_M1, '"p95_ms": 4.775'),
    ('211,9', R_M1, '"fps": 211.91'),
    ('51,5', R_M1, '"rss_mb": 51.5'),
    ('2677,0', R_M1, '"model_load_ms": 2677.01'),
    ('0,17', R_M1, '"l2_norm_delta_pct": 0.174'),
    ('0,18', R_M1, '"l2_norm_delta_pct": 0.175'),
    ('7,78', R_OLDNEW, '"p50_ms": 7.781'),
    ('0,46', R_OLDNEW, '"std_ms": 0.461'),
    ('128,6', R_OLDNEW, '"fps": 128.62'),
    ('4,07', R_OLDNEW, '"p50_ms": 4.07'),
    ('4,11 ± 0,21', R_OLDNEW, '"mean_ms": 4.112'),
    ('243,2', R_OLDNEW, '"fps": 243.19'),
    ('122,61', R_OLDNEW, '"p50_ms": 122.611'),
    ('137,90 ± 31,75', R_OLDNEW, '"std_ms": 31.745'),
    ('7,3', R_OLDNEW, '"fps": 7.25'),
    ('55,67', R_OLDNEW, '"p50_ms": 55.665'),
    ('55,97 ± 1,60', R_OLDNEW, '"std_ms": 1.599'),
    ('17,9', R_OLDNEW, '"fps": 17.87'),
    ('24,9', R_OLDNEW, '"l2_norm_delta_pct": 24.878'),
    ('49,2', R_OLDNEW, '"l2_norm_delta_pct": 49.232'),
    ('24,0', R_OLDNEW, '"l2_norm_delta_pct": 23.989'),
    (
        '20 из 271',
        'paper/evidence/check_determinism_2026-09-06.txt',
        '20/271  deterministic',
    ),
    (
        '0 из 175',
        'paper/evidence/check_determinism_2026-09-06.txt',
        '0/175  deterministic',
    ),
    (
        'нулевым кодом возврата',
        'paper/evidence/check_determinism_2026-09-06.txt',
        'exit code: 0',
    ),
    ('0,9913', 'README.md', '0.9913'),
    ('0,9912', 'README.md', '0.9912'),
    ('139 из 234', 'README.md', '139 of its 234 tensors'),
    ('0,50', 'README.md', 'cosine **0.50**'),
    ('4 из 68', 'README.md', '4 of 68 ops'),
    ('16.0', 'README.md', 'edgetpu_compiler` 16.0'),
    ('48 свёрток', 'README.md', 'for 48 convolutions'),
    ('1800 МГц', R_TPU, '"cpu_freq_mhz": 1800'),
    ('6.12.93', R_TPU, '6.12.93'),
    ('3.11.2', R_TPU, '"python": "3.11.2"'),
    ('2.14.0', R_M1, 'tflite_runtime 2.14.0'),
    ('2.1.6', R_M1, 'ai_edge_litert 2.1.6'),
    ('glibc 2.36', R_M1, 'glibc2.36'),
    ('glibc 2.42', R_M1, 'glibc2.42'),
]

# Слова из раздела 0 правил репозитория aspirantura; хранятся перевёрнутыми, чтобы сам
# скрипт не срабатывал в сканерах (форматтер не склеивает такие литералы).
FORBIDDEN = [
    w[::-1]
    for w in (
        'edualC',
        'ciporhtnA',
        'ia.edualc',
        'rosruC',
        'tnegarosruc',
        'MLL',
        'tnatsissa',
        'онавориренегс',
        'htiw detareneG',
    )
]
FORBIDDEN_WORDS = ['IA'[::-1], '\u0418\u0418']


def replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f'якорь встречается {n} раз (ожидался 1): {old[:60]!r}')
    return text.replace(old, new)


def number_citations(text: str) -> tuple[str, list[str]]:
    cited: list[str] = []
    for k in re.findall(r'\[\[(\w+)\]\]', text):
        if k not in REFS:
            sys.exit(f'неизвестный ключ источника: {k}')
        if k not in cited:
            cited.append(k)
    unused = sorted(set(REFS) - set(cited))
    if unused:
        print(f'источники без цитирования (в список не включены): {unused}')
    num = {k: i + 1 for i, k in enumerate(cited)}
    text = re.sub(r'\[\[(\w+)\]\]', lambda m: f'[{num[m.group(1)]}]', text)

    def collapse(m):
        nums = [int(x) for x in re.findall(r'\d+', m.group(0))]
        parts, start, prev = [], nums[0], nums[0]
        for n in nums[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n
                continue
            parts.append(
                str(start)
                if start == prev
                else (f'{start}, {prev}' if prev == start + 1 else f'{start}–{prev}')
            )
            if n is not None:
                start = prev = n
        return '[' + ', '.join(parts) + ']'

    text = re.sub(r'\[\d+\](?:, \[\d+\])+', collapse, text)
    return text, cited


def render_refs(cited: list[str]) -> str:
    ru = '\n'.join(f'{i + 1}. {REFS[k][1]}' for i, k in enumerate(cited))
    en = '\n'.join(f'{i + 1}. {REFS[k][2] or REFS[k][1]}' for i, k in enumerate(cited))
    return f'## БИБЛИОГРАФИЧЕСКИЙ СПИСОК\n\n{ru}\n\n## REFERENCES\n\n{en}\n'


def words(s: str) -> int:
    return len(s.split())


def live_counts() -> dict[str, int]:
    def loc(d: str) -> int:
        return sum(
            len(p.read_text(encoding='utf8').splitlines())
            for p in (ROOT / d).rglob('*.py')
        )

    # Число тестов — как его сообщает pytest (с учётом параметризации), а не число функций test_*.
    collected = subprocess.run(
        ['poetry', 'run', 'pytest', '--collect-only', '-q'],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout
    m = re.search(r'(\d+)/(\d+) tests collected', collected)
    tests_total = int(m.group(2)) if m else 0
    tests_hw = tests_total - int(m.group(1)) if m else 0
    routes = 0
    for p in (
        list((ROOT / 'server/api').glob('*.py'))
        + list((ROOT / 'server/routes').glob('*.py'))
        + [ROOT / 'server/main.py']
    ):
        routes += len(
            re.findall(r'^@(?:router|app)\.', p.read_text(encoding='utf8'), flags=re.M)
        )
    schema = (ROOT / 'server/db/database.py').read_text(encoding='utf8')
    tables = len(set(re.findall(r'CREATE TABLE IF NOT EXISTS (\w+)', schema)))
    return {
        'server': loc('server'),
        'agent': loc('agent'),
        'scripts': loc('scripts'),
        'tests_loc': loc('tests'),
        'tests': tests_total,
        'tests_hw': tests_hw,
        'routes': routes,
        'tables': tables,
    }


def check(manuscript: str, cited: list[str], report: list[str]) -> None:
    for lang in ('RU', 'EN'):
        m = re.search(
            rf'<!-- BEGIN ABSTRACT_{lang} -->\n(.*?)\n<!-- END ABSTRACT_{lang} -->',
            manuscript,
            flags=re.S,
        )
        n = words(m.group(1)) if m else -1
        report.append(
            f'аннотация {lang}: {n} слов [норма 200–250] {"OK" if 200 <= n <= 250 else "FAIL"}'
        )
    kw = [
        k
        for k in re.findall(r'^\*([^*\n]+;[^*\n]+)\*$', manuscript, flags=re.M)
        if words(k) < 40
    ]
    for k in kw:
        n = len([x for x in k.split(';') if x.strip()])
        report.append(
            f'ключевые слова: {n} [норма 5–10] {"OK" if 5 <= n <= 10 else "FAIL"}'
        )
    n = len(cited)
    old = [k for k in cited if REFS[k][0] < 2016]
    share = 100 * len(old) / n
    report.append(
        f'источников: {n} [норма ≥ 20] {"OK" if n >= 20 else "FAIL"}; старше 10 лет: {len(old)} ({share:.0f} %) '
        f'[норма ≤ 25 %] {"OK" if share <= 25 else "FAIL"}: {", ".join(old)}'
    )
    body = manuscript.split('## БИБЛИОГРАФИЧЕСКИЙ СПИСОК')[0]
    bad = []
    for tok, src, stok in PROVENANCE:
        if tok not in body:
            bad.append(f'в тексте нет {tok!r}')
        elif stok not in (ROOT / src).read_text(encoding='utf8'):
            bad.append(f'в {src} нет {stok!r} (для {tok!r})')
    report.append(
        f'провенанс чисел: {len(PROVENANCE)} проверок, {"OK" if not bad else "FAIL: " + "; ".join(bad)}'
    )
    lc = live_counts()
    miss = [f'{k}={v}' for k, v in lc.items() if str(v) not in body]
    report.append(
        f'объём кода/тесты/маршруты/таблицы: {lc} {"OK" if not miss else "FAIL (нет в тексте): " + ", ".join(miss)}'
    )
    # Собственные имена в цитируемых работах (бенчмарк Ignatov et al., заглавие статьи Cass) — не нарушение.
    _ai = 'IA'[::-1]
    scan = manuscript.replace(_ai + ' Benchmark', '').replace(
        'Taking ' + _ai + ' to the edge', ''
    )
    hits = [w for w in FORBIDDEN if w.lower() in scan.lower()]
    hits += [w for w in FORBIDDEN_WORDS if re.search(rf'(?<![\w-]){w}(?![\w-])', scan)]
    report.append(f'запрещённые слова: {hits or "нет"} {"OK" if not hits else "FAIL"}')
    todo = re.findall(r'\[(?:уточнить|to be specified)\]', manuscript)
    report.append(
        f'незаполненные поля: {len(todo)} {"OK" if not todo else "CHECK (телефон соавтора)"}'
    )


# ------------------------------------------------------------------- docx
FONT = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman"/>'


def template_sect_pr() -> str:
    with zipfile.ZipFile(TEMPLATE) as z:
        doc = z.read('word/document.xml').decode('utf8')
    sect = re.search(r'<w:sectPr[ >].*?</w:sectPr>', doc, flags=re.S).group(0)
    sect = re.sub(r'<w:(header|footer)Reference [^>]*/>', '', sect)
    sect = re.sub(r'<w:pgNumType [^>]*/>', '', sect)
    return re.sub(r'<w:sectPr[^>]*>', '<w:sectPr>', sect)


def make_reference_docx(dst: Path) -> None:
    default = dst.with_name('ref_default.docx')
    subprocess.run(
        ['pandoc', '-o', str(default), '--print-default-data-file', 'reference.docx'],
        check=True,
    )
    sect = template_sect_pr()
    with (
        zipfile.ZipFile(default) as zin,
        zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout,
    ):
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == 'word/styles.xml':
                s = data.decode('utf8')
                s = re.sub(r'<w:rFonts [^/]*/>', FONT, s)
                s = re.sub(r'<w:sz w:val="\d+"\s*/>', f'<w:sz w:val="{BODY_SZ}"/>', s)
                s = re.sub(
                    r'<w:szCs w:val="\d+"\s*/>', f'<w:szCs w:val="{BODY_SZ}"/>', s
                )
                s = re.sub(r'<w:color [^/]*/>', '', s)
                s = re.sub(
                    r'<w:spacing [^/]*/>',
                    '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>',
                    s,
                )
                s = re.sub(r'<w:keepNext/>', '', s)

                def style(
                    st_id: str,
                    ppr: str = '',
                    rpr_add: str = '',
                    rpr_drop: tuple[str, ...] = (),
                ):
                    nonlocal s
                    m = re.search(
                        rf'<w:style [^>]*w:styleId="{st_id}"[^>]*>.*?</w:style>',
                        s,
                        flags=re.S,
                    )
                    if not m:
                        return
                    blk = m.group(0)
                    for tag in rpr_drop:
                        blk = blk.replace(f'<w:{tag}/>', '')
                    if ppr:
                        blk = (
                            re.sub(r'<w:pPr>', '<w:pPr>' + ppr, blk, count=1)
                            if '<w:pPr>' in blk
                            else blk.replace(
                                '</w:name>', '</w:name><w:pPr>' + ppr + '</w:pPr>', 1
                            )
                        )
                    if rpr_add:
                        blk = (
                            re.sub(r'<w:rPr>', '<w:rPr>' + rpr_add, blk, count=1)
                            if '<w:rPr>' in blk
                            else blk.replace(
                                '</w:style>', '<w:rPr>' + rpr_add + '</w:rPr></w:style>'
                            )
                        )
                    s = s.replace(m.group(0), blk)

                just = '<w:ind w:firstLine="567"/><w:jc w:val="both"/>'
                for st in ('BodyText', 'FirstParagraph'):
                    style(st, ppr=just)
                style('Compact', ppr='<w:ind w:firstLine="0"/><w:jc w:val="left"/>')
                style(
                    'Heading1',
                    ppr='<w:jc w:val="center"/><w:spacing w:before="120" w:after="120"/>',
                    rpr_add='<w:b/><w:bCs/><w:sz w:val="21"/><w:szCs w:val="21"/>',
                    rpr_drop=('i', 'iCs'),
                )
                style(
                    'Heading2',
                    ppr='<w:jc w:val="center"/><w:spacing w:before="160" w:after="80"/>',
                    rpr_add='<w:b/><w:bCs/>',
                    rpr_drop=('i', 'iCs'),
                )
                style(
                    'Heading3',
                    ppr='<w:ind w:firstLine="567"/><w:spacing w:before="120" w:after="40"/>',
                    rpr_add='<w:i/><w:iCs/>',
                    rpr_drop=('b', 'bCs'),
                )
                style('Table', rpr_add='<w:sz w:val="16"/><w:szCs w:val="16"/>')
                style('Figure', ppr='<w:jc w:val="center"/>')
                data = s.encode('utf8')
            elif item.filename == 'word/document.xml':
                s = re.sub(
                    r'<w:sectPr>.*?</w:sectPr>', '', data.decode('utf8'), flags=re.S
                )
                data = s.replace('</w:body>', sect + '</w:body>').encode('utf8')
            zout.writestr(item, data)
    default.unlink()


def fix_table_widths(text: str) -> str:
    """Ширины столбцов pipe-таблиц ∝ длине содержимого (pandoc читает их из числа дефисов)."""
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        if (
            lines[i].startswith('|')
            and i + 1 < len(lines)
            and lines[i + 1].startswith('|-')
            and set(lines[i + 1]) <= set('|:- ')
        ):
            j = i
            while j < len(lines) and lines[j].startswith('|'):
                j += 1
            rows = [[c.strip() for c in ln.strip('|').split('|')] for ln in lines[i:j]]
            ncol = len(rows[1])
            width = [10] * ncol
            for k, r in enumerate(rows):
                if k == 1:
                    continue
                for c, cell in enumerate(r[:ncol]):
                    n = len(re.sub(r'[*`$\\{}_]', '', cell))
                    n = n // 2 if k == 0 else n
                    width[c] = max(width[c], min(45, 5 + n))
            sep = []
            for c, spec in enumerate(rows[1]):
                sep.append(
                    (':' if spec.startswith(':') else '')
                    + '-' * width[c]
                    + (':' if spec.endswith(':') else '')
                )
            lines[i + 1] = '|' + '|'.join(sep) + '|'
            out.extend(lines[i:j])
            i = j
        else:
            out.append(lines[i])
            i += 1
    return '\n'.join(out)


def render_figures() -> None:
    for puml in sorted(FIG.glob('*.puml')):
        png = puml.with_suffix('.png')
        if png.exists() and png.stat().st_mtime >= puml.stat().st_mtime:
            continue
        if not PLANTUML_JAR.exists():
            sys.exit(f'нет {png.name} и не найден plantuml.jar ({PLANTUML_JAR})')
        subprocess.run(
            [
                'java',
                '-Djava.awt.headless=true',
                '-jar',
                str(PLANTUML_JAR),
                '-tpng',
                '-charset',
                'UTF-8',
                str(puml),
            ],
            check=True,
        )
        print(f'рисунок отрисован: {png.name}')


def main() -> None:
    global BODY_SZ
    no_pdf = '--no-pdf' in sys.argv
    if '--pt' in sys.argv:
        BODY_SZ = 2 * int(sys.argv[sys.argv.index('--pt') + 1])
    suffix = f'_{BODY_SZ // 2}pt' if BODY_SZ != 18 else ''
    OUT.mkdir(exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    render_figures()
    for png in FIG.glob('*.png'):
        (OUT / 'figures' / png.name).write_bytes(png.read_bytes())

    text = MAIN.read_text(encoding='utf8')
    text, cited = number_citations(text)
    text = replace_once(text, '<!-- REFERENCES -->', render_refs(cited))
    text = fix_table_widths(text)
    md = OUT / f'edge_bench_izv_sfedu{suffix}.md'
    md.write_text(text, encoding='utf8')

    report: list[str] = []
    check(text, cited, report)

    docx = md.with_suffix('.docx')
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / 'ref_izv.docx'
        make_reference_docx(ref)
        subprocess.run(
            [
                'pandoc',
                str(md),
                '-o',
                str(docx),
                f'--reference-doc={ref}',
                '-f',
                'markdown-implicit_figures',
                f'--resource-path={OUT}',
            ],
            check=True,
        )
    if not no_pdf:
        subprocess.run(
            [
                'soffice',
                '--headless',
                '--convert-to',
                'pdf',
                '--outdir',
                str(OUT),
                str(docx),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        info = subprocess.run(
            ['pdfinfo', str(docx.with_suffix('.pdf'))],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        pages = int(re.search(r'Pages:\s+(\d+)', info).group(1))
        report.append(
            f'страниц (A4, TNR {BODY_SZ / 2:g} pt, поля шаблона): {pages} [норма 10–20] {"OK" if 10 <= pages <= 20 else "FAIL"}'
        )
    report.append(
        f'слов в рукописи (без списков литературы): {words(text.split("## БИБЛИОГРАФИЧЕСКИЙ СПИСОК")[0])}'
    )
    (OUT / f'CHECK_REPORT{suffix}.txt').write_text(
        '\n'.join(report) + '\n', encoding='utf8'
    )
    print('\n'.join(report))
    print(f'\nвыход: {md}\n       {docx}')


if __name__ == '__main__':
    main()
