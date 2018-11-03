python -u train.py --lr 0.02 --momentum 0.5 --num_hidden 2 --sizes 100,100 --activation sigmoid --loss sq --opt adam --batch_size 20 --anneal true --save_dir q7/nag/model --expt_dir q7/nag/log --train data/train.csv --test data/test.csv --val data/val.csv --epochs 20

