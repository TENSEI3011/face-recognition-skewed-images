fixups = [
    ('\u2014', ' - '),
    ('\u2013', '-'),
    ('\u2019', "'"),
    ('\u2018', "'"),
    ('\u201c', '"'),
    ('\u201d', '"'),
    ('\u2022', '*'),
    ('\u2713', 'OK'),
    ('\u26a0', '!'),
    ('\u2717', 'X'),
]

for fname in ['generate_setup_pdf.py', 'generate_army_guide_pdf.py']:
    txt = open(fname, encoding='utf-8').read()
    for bad, good in fixups:
        txt = txt.replace(bad, good)
    open(fname, 'w', encoding='utf-8').write(txt)
    print(f'Fixed: {fname}')
