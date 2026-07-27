import re, sys, collections
log = open(sys.argv[1], errors='replace').read().splitlines()
cur = []
groups = collections.OrderedDict()
for line in log:
    if line.startswith('STCDP_FINAL_TRANSFER'):
        cur.append(line)
    elif line.startswith('STCDP_FINAL_END'):
        name = line.split('sdsc=')[1].strip()
        groups[name] = cur; cur = []
    elif line.startswith('STCDP_FINAL_BEGIN'):
        cur = []
print(f"{'sdsc':<28}{'xfers':>6}{'local':>7}{'remote':>8}{'src_cores':>10}{'dst_cores':>10}{'local_MiB':>11}{'remote_MiB':>11}")
for name, lines in groups.items():
    loc=rem=0; lb=rb=0; sc=set(); dc=set()
    for l in lines:
        d = dict(re.findall(r'(\w+)=([-\w\[\]]+)', l))
        b = int(d.get('logical_bytes', 0))
        if d.get('remote') == '1': rem += 1; rb += b
        else: loc += 1; lb += b
        sc.add(d.get('src_core')); dc.add(d.get('dst_core'))
    print(f"{name:<28}{len(lines):>6}{loc:>7}{rem:>8}{len(sc):>10}{len(dc):>10}{lb/1048576:>11.2f}{rb/1048576:>11.2f}")
