from pathlib import Path


def main():
    rows = [
        ('A0', 'rtdetrv2_r50vd_6x_visdrone_p2', 'baseline'),
        ('A1', 'rtdetrv2_r50vd_6x_visdrone_p2_dq_a1_count', 'CCM + Count Loss'),
        ('A2', 'rtdetrv2_r50vd_6x_visdrone_p2_dq_a2_cgfe', 'CCM + CGFE'),
        ('A3', 'rtdetrv2_r50vd_6x_visdrone_p2_dq_a3_dynamic_query', 'CCM + Dynamic Query'),
        ('A4', 'rtdetrv2_r50vd_6x_visdrone_p2_dq_a4_full', 'Full'),
    ]
    print('| Exp | Output | Note | Last log line |')
    print('|---|---|---|---|')
    for exp, output, note in rows:
        log = Path('output') / output / 'log.txt'
        last = ''
        if log.exists():
            lines = [line.strip() for line in log.read_text(errors='ignore').splitlines() if line.strip()]
            last = lines[-1] if lines else ''
        print(f'| {exp} | {output} | {note} | {last} |')


if __name__ == '__main__':
    main()
