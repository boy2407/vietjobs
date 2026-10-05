import json,sys
d=json.load(open('batch10.json'))
a,b=int(sys.argv[1]),int(sys.argv[2])
for i in range(a,min(b,len(d))):
    x=d[i]
    print(f"[{i}] row={x['row']} | {x['job_title']}\n D: {(x['description'] or '')[:400]}\n R: {(x['requirements'] or '')[:200]}\n".replace('\n\n','\n'))
