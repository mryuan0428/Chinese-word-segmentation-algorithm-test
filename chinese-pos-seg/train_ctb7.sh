CUDA_VISIBLE_DEVICES=2 nohup python tagger.py train -iter 40 -tb 64 -ed 64 -p CTB7 -t train.txt -bt 10 -d dev.txt -cp -m CTB7 -emb Embeddings/glove.txt > ctb7.log &
