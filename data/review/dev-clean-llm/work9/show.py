import json,sys
d=json.load(open('batch9.json'))
a,b=int(sys.argv[1]),int(sys.argv[2])
for i,x in enumerate(d[a:b],a):
    print(f"#{i} row={x['row']} | {x['job_title']}\n D: {(x['description'] or '')[:400]!r}\n R: {(x['requirements'] or '')[:200]!r}\n")
