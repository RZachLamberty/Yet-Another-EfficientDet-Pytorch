# testing speed of this repo only on a single gpu instance

+ create `p3.2xlarge` (or whatever)

```shell
git clone https://github.com/RZachLamberty/Yet-Another-EfficientDet-Pytorch.git
cd ~/Yet-Another-EfficientDet-Pytorch

# setup
./scripts/sb_init_dev.sh
./scripts/get_data.py

# test results
rm -f test_results.txt
python ./scripts/sb_speed_test.py -c 0 --gpu >> test_results.txt

# if the aboved work, try with similar batch sizes for c=0
python ./scripts/sb_speed_test.py -c 0 --gpu --image-batch-size 4 >> test_results.txt
python ./scripts/sb_speed_test.py -c 0 --gpu --image-batch-size 8 >> test_results.txt
python ./scripts/sb_speed_test.py -c 0 --gpu --image-batch-size 16 >> test_results.txt

# choose a batch that works, try for different c values
python ./scripts/sb_speed_test.py -c 1 --gpu >> test_results.txt
python ./scripts/sb_speed_test.py -c 2 --gpu >> test_results.txt
python ./scripts/sb_speed_test.py -c 3 --gpu >> test_results.txt
python ./scripts/sb_speed_test.py -c 4 --gpu >> test_results.txt
python ./scripts/sb_speed_test.py -c 5 --gpu >> test_results.txt
python ./scripts/sb_speed_test.py -c 6 --gpu >> test_results.txt
python ./scripts/sb_speed_test.py -c 7 --gpu >> test_results.txt
python ./scripts/sb_speed_test.py -c 8 --gpu >> test_results.txt

# if that worked, then with float16
python ./scripts/sb_speed_test.py -c 0 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 1 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 2 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 3 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 4 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 5 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 6 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 7 --gpu --float16 >> test_results.txt
python ./scripts/sb_speed_test.py -c 8 --gpu --float16 >> test_results.txt 
```

+ on local machine, `scp edtest:~/Yet-Another-EfficientDet-Pytorch/test_results.txt .`
+ shut down instance


parsing the `test_results.txt` file:

```python
import re
import numpy as np
import pandas as pd

with open('data/test_results.txt', 'r') as fp:
    results = [_ for _ in fp.read().split('=' * 80) if _]

def parse_results(r):
    d = {}
    d['compound_coef'] = int(re.search('compound_coef = (\d)', r, re.MULTILINE).groups()[0])
    d['nms_threshold'] = float(re.search('nms_threshold = ([\d\.]+)', r, re.MULTILINE).groups()[0])
    d['gpu'] = bool(re.search('gpu = (True|False)', r, re.MULTILINE).groups()[0])
    d['float16'] = bool(re.search('float16 = (True|False)', r, re.MULTILINE).groups()[0])
    d['image_batch_size'] = int(re.search('image_batch_size = (\d+)', r, re.MULTILINE).groups()[0])
    fps_spf = re.findall('batch_time.*\nbatch_size.*\nFPS = (?P<fps>[\d\.]+)\nSPF = (?P<spf>[\d\.]+)',
                         r, re.MULTILINE)
    d['batch_fps_vals'] =  [float(fps) for (fps, spf) in fps_spf]
    d['batch_spf_vals'] =  [float(spf) for (fps, spf) in fps_spf]
    
    for k in ['fps', 'spf']:
        d[f"batch_{k}_avg"] = np.mean(d[f"batch_{k}_vals"]) 
        d[f"batch_{k}_std"] = np.std(d[f"batch_{k}_vals"])
    
    try:
        t, fps, spf = re.search('final summary.*\ntotal processing time: (?P<total_time>[\d\.]+) \(s\)\n.*\n.*\nFPS: (?P<fps>[\d\.]+)\nSPF: (?P<spf>[\d\.]+)',
                                r, re.MULTILINE).groups()
        d['total_time'] = float(t)
        d['fps'] = float(fps)
        d['spf'] = float(spf)
    except AttributeError:
        d['fps'] = None
        d['spf'] = None

    return d

results = pd.DataFrame([parse_results(r) for r in results])
```
