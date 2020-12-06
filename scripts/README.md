# testing speed of this repo only on a single gpu instance

+ create `p3.2xlarge` (or whatever)
+ run `sb_init_dev.sh`
+ run `get_data.py`
+ run `sb_speed_test.py`
 
```shell
cd ~/Yet-Another-EfficientDet-Pytorch
rm -f test_results.txt
python ./scripts/sb_speed_test.py -c 0 --gpu >> test_results.txt
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