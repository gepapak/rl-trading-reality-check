import sys,re
from pathlib import Path
aid=sys.argv[1]
t=Path('text',aid+'.txt').read_text(encoding='utf-8')
print(f'\n######## {aid} | {t[:110]}')
P={'LIQ':r'price[- ]taker|market impact|price impact|liquidit|order book|market depth|does not (affect|influence) the (market )?price|small (amount|volume)s?',
   'COST':r'transaction cost|trading fee|fees?\b|degradation cost',
   'SEED':r'random seeds?|\bseeds?\b|independent runs|repeated \w+ times|averaged over \w+ (runs|trainings|policies)',
   'CLAIM':r'outperform\w*|improv\w* (the )?(average )?(daily )?profit|more (net )?profit|increase[sd]? (the )?profit',
   'SPIKE':r'price spikes?|outliers?|winsori|remove[sd]? (extreme|outlier)|cap(ped)? (the )?prices?',
   'SOLV':r'bankrupt|\bruin\b|margin call|collateral|capital (constraint|requirement)|credit limit'}
for k,p in P.items():
    hs=[m for m in re.finditer(p,t,flags=re.I) if not (k=='SPIKE' and 'clip' in t[m.start()-5:m.end()+5].lower())]
    print(f'[{k}] {len(hs)}')
    for m in hs[:2]:
        print('   ...'+t[max(0,m.start()-170):m.end()+170]+'...')
