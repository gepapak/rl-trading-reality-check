import sys,re
from pathlib import Path
aid=sys.argv[1]; pats=sys.argv[2:]
t=Path('text',aid+'.txt').read_text(encoding='utf-8')
print(f'######## {aid} | {t[:150]}')
for p in pats:
    hs=list(re.finditer(p,t,flags=re.I))
    print(f'\n[{p}] {len(hs)} hits')
    for m in hs[:4]:
        print('   ...'+t[max(0,m.start()-200):m.end()+230]+'...')
