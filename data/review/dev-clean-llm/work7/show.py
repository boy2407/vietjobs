import json,sys
d=json.load(open('batch7.json'))
a,b=int(sys.argv[1]),int(sys.argv[2])
for i,x in enumerate(d[a:b],a):
    print(f"#{i} row={x['row']} | {x['job_title']}\nD: {(x['description'] or '')[:400]}\nR: {(x['requirements'] or '')[:200]}\n".replace('\n\n','\n'))
