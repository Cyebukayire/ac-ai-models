import sys
from datetime import datetime

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def process_ptf_file(inp, out):
    print('In:', inp, 'Out:', out, flush=True)
    print(time_now(), f'Reading input file: {inp} ...', flush=True)
    data = []
    with open(inp, 'r') as inpf:
        for line in inpf:
            if len(line) > 1:
                idx = line.index('\t')
                val = float(line[:idx])
                data.append((val,line[(idx+1):]))
    print(time_now(), f'Sorting {len(data)} lines ...', flush=True)
    data.sort(key=lambda x: x[0], reverse=False)
    print(time_now(), f'Writing output file: {out} ...', flush=True)
    with open(out, "w", encoding='utf-8') as outf:
        for val,txt in data:
            outf.write(txt)
    print(time_now(), f'Done!')

if __name__ == '__main__':
    args = sys.argv[1:]
    inp = args[0]
    out = args[1]
    process_ptf_file(inp, out)
