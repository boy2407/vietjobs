import json
codes=dict(l.split('=') for l in open('work16/codes.txt').read().split())
lab={}
for l in open('work16/labels.txt',encoding='utf-8'):
    l=l.strip()
    if not l: continue
    r,b,o,why=l.split('|')
    assert int(r) not in lab, r
    lab[int(r)]=(b,o,why)
d=json.load(open('batch16.json'))
out=[]
for x in d:
    b,o,why=lab[x['row']]
    alts=[codes[b]]+[codes[c] for c in o.split(',') if c]
    out.append({'row':x['row'],'nhan_tot_nhat':codes[b],'cac_nhan_hop_ly':alts,'ly_do':why})
assert len(out)==len(d)==len(lab), (len(out),len(d),len(lab))
json.dump(out,open('out16.json','w'),ensure_ascii=False,indent=1)
print(len(out))
